import os, re, math, torch, tiktoken, json, time
from functools import partial
from torch.utils.data import DataLoader
import torch.nn as nn
import atexit
# --- TENSORBOARD IMPORT ---
from torch.utils.tensorboard import SummaryWriter
import torch.cuda
# --------------------------

# --- IMPORTS FROM OTHER FILES (Assumed to be in Python path) ---
from Instructional.Training.functions import train_model_simple
from Instructional.model import GPTModel, CHOOSE_MODEL, BASE_CONFIG
from Instructional.Data.data_set import InstructionDataset
from Instructional.Data.collate import custom_collate_fn
from Instructional.Data.format import format_input
from Instructional.Accuracy.post_training import post_training_generate
# -------------------------------------------------------------

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ModuleNotFoundError:
    print("Running outside Colab — skipping Google Drive mount.")

# --- CONFIGURATION FOR TENSORBOARD RUNS ---
BASE_EXPERIMENT_NAME = f"LLM_FineTuning_Comparison"
STAGE1_RUN_NAME = "Full_FineTuning_SFT"
STAGE2_RUN_NAME = "LoRA_PEFT_Final_Tune"
# ---------------------- Device & Seed ----------------------
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Tokenizer ----------------------
tokenizer = tiktoken.get_encoding("gpt2")

# ---------------------- LoRA Layer ----------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scaling = self.alpha / self.r
    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias) + \
               self.scaling * nn.functional.linear(x, self.B @ self.A)

base_dir = "/content/drive/MyDrive/Finetuned_checkpoints"
os.makedirs(base_dir, exist_ok=True)

# Checkpoint paths
stage1_checkpoint_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft-stage1.pth"
stage1_checkpoint_path = os.path.join(base_dir, stage1_checkpoint_name)
final_checkpoint_name = "model.pth"
final_checkpoint_path = os.path.join(base_dir, final_checkpoint_name)

# --- GLOBAL MODEL/OPTIMIZER DEFINITION (FIX for NameError in atexit) ---
model = None 
optimizer = None 
best_val_loss = float('inf')

def backup_save():
    global model, optimizer, best_val_loss 
    if model is not None and optimizer is not None:
        print("\n❗ Detected interruption. Saving backup checkpoint...")
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, stage1_checkpoint_path)
            print(f"✅ Backup saved to: {stage1_checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Could not save backup state: {e}")

atexit.register(backup_save)

# ---------------------- Load Dataset ----------------------
try:
    with open("train_data.json", "r") as f:
        train_data_json = json.load(f)
    with open("val_data.json", "r") as f:
        val_data_json = json.load(f)
    with open("test_data.json", "r") as f:
        test_data_json = json.load(f)
    print("✅ Successfully loaded data from JSON files.")
except FileNotFoundError:
    raise FileNotFoundError("Data JSON files not found. Please run data_setup.py first to create them.")

train_dataset = InstructionDataset(train_data_json, tokenizer)
val_dataset = InstructionDataset(val_data_json, tokenizer)
test_dataset = InstructionDataset(test_data_json, tokenizer)
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=0, collate_fn=customized_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0, collate_fn=customized_collate_fn)
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")


# -----------------------------------------------------------
# 🔹 The Two-Stage Training Logic
# -----------------------------------------------------------

if not os.path.exists(stage1_checkpoint_path):
    # ---------------------- STAGE 1: Full Fine-Tuning ----------------------
    print("\n--- Starting Stage 1: Full Fine-Tuning ---")
    
    # Initialize Writer for STAGE 1
    log_path_s1 = os.path.join("runs", BASE_EXPERIMENT_NAME, STAGE1_RUN_NAME)
    writer = SummaryWriter(log_path_s1)
    
    model = GPTModel(BASE_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    num_epochs_stage1 = 2
    best_val_loss = float('inf')

    try:
        train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs_stage1, eval_freq=50, eval_iter=5,
            start_context=format_input(val_data_json[0]), tokenizer=tokenizer,
            checkpoint_path=stage1_checkpoint_path,
            grad_accum_steps=4,
            best_val_loss=best_val_loss,
            writer=writer 
        )
        print(f"✅ Stage 1 training completed. Final checkpoint saved to {stage1_checkpoint_path}.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❗ Stage 1 stopped due to out-of-memory error. Using latest checkpoint for Stage 2.")
        else:
            raise
    finally:
        writer.close()
    
else:
    print(f"\n--- Stage 1 checkpoint found ({stage1_checkpoint_path}). Skipping to Stage 2 ---")
    
# ---------------------- STAGE 2: LoRA Fine-Tuning ----------------------
print("\n--- Starting Stage 2: LoRA Fine-Tuning ---")

# Initialize Writer for STAGE 2
log_path_s2 = os.path.join("runs", BASE_EXPERIMENT_NAME, STAGE2_RUN_NAME)
writer = SummaryWriter(log_path_s2)

model = GPTModel(BASE_CONFIG).to(device)

try:
    # --- FIX: strict=False added to bypass corrupt LoRA keys in Stage 1 checkpoint ---
    model.load_state_dict(
        torch.load(stage1_checkpoint_path, map_location=device)['model_state_dict'],
        strict=False
    )
    print(f"✅ Loaded Stage 1 checkpoint from {stage1_checkpoint_path} for LoRA fine-tuning (ignoring extra keys).")
except FileNotFoundError:
    print(f"Error: Stage 1 checkpoint not found at {stage1_checkpoint_path}. Exiting.")
    exit()

# Apply LoRA layers
for block in model.trf_blocks:
    if isinstance(block.att.W_query, nn.Linear):
        block.att.W_query = LoRALinear(block.att.W_query.in_features, block.att.W_query.out_features)
    if isinstance(block.att.W_key, nn.Linear):
        block.att.W_key = LoRALinear(block.att.W_key.in_features, block.att.W_key.out_features)
    if isinstance(block.att.W_value, nn.Linear):
        block.att.W_value = LoRALinear(block.att.W_value.in_features, block.att.W_value.out_features)
    if isinstance(block.att.out_proj, nn.Linear):
        block.att.out_proj = LoRALinear(block.att.out_proj.in_features, block.att.out_proj.out_features)

# Freeze all original parameters
for name, param in model.named_parameters():
    if "A" not in name and "B" not in name:
        param.requires_grad = False

# Re-initialize the optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=1e-3, weight_decay=0.01)

num_epochs_stage2 = 10
best_val_loss = float('inf')

try:
    train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs_stage2, eval_freq=50, eval_iter=5,
        start_context=format_input(val_data_json[0]), tokenizer=tokenizer,
        checkpoint_path=final_checkpoint_path,
        grad_accum_steps=4,
        best_val_loss=best_val_loss,
        writer=writer 
    )
    print(f"✅ Stage 2 training completed. Final checkpoint saved to {final_checkpoint_path}.")
except Exception as e:
    print(f"An error occurred during Stage 2 training: {e}")
    exit()
finally:
    if writer is not None:
        writer.close() 

# -----------------------------------------------------------
# 🔹 Post-Training Generation and Evaluation
# -----------------------------------------------------------
final_val_loss = float('inf')
try:
    final_checkpoint = torch.load(final_checkpoint_path, map_location=device)
    final_val_loss = final_checkpoint.get('best_val_loss', float('inf'))
    model.load_state_dict(final_checkpoint['model_state_dict'])
    print(f"✅ Loaded final checkpoint for post-training generation.")
except FileNotFoundError:
    print(f"Error: Final checkpoint not found at {final_checkpoint_path}. Exiting.")
    exit()

output_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-responses.json"
output_path = os.path.join(base_dir, output_name)
test_data_json = post_training_generate(model, tokenizer, device, test_data_json)
with open(output_path, "w") as f:
    json.dump(test_data_json, f, indent=4)
print(f"Responses saved at {output_path}")