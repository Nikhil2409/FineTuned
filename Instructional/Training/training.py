import os, re, time, math, torch, tiktoken, json
from functools import partial
from torch.utils.data import DataLoader
import torch.nn as nn
import atexit
from google.colab import drive
from Instructional.Training.functions import train_model_simple
from Instructional.Training.loss import calc_loss_batch, calc_loss_loader
from Instructional.model import GPTModel, CHOOSE_MODEL, BASE_CONFIG
from Instructional.Data.data_set import InstructionDataset
from Instructional.Data.collate import custom_collate_fn
from Instructional.Data.format import format_input
from Instructional.Accuracy.post_training import post_training_generate

# PEFT LoRA imports
from peft import PeftModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

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

# ---------------------- Checkpoint Paths ----------------------
base_dir = "/content/drive/MyDrive/Finetuned_checkpoints"
os.makedirs(base_dir, exist_ok=True)

# Checkpoint for the end of Stage 1, now also used as the backup
stage1_checkpoint_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft-stage1.pth"
stage1_checkpoint_path = os.path.join(base_dir, stage1_checkpoint_name)

# Final checkpoint for the end of Stage 2 (LoRA)
final_checkpoint_name = "model.pth"
final_checkpoint_path = os.path.join(base_dir, final_checkpoint_name)

# ---------------------- Helper Functions (Included for self-containment) ----------------------

# This function calculates the loss for a single batch.
def calc_loss_batch(outputs, targets):
    outputs = outputs.view(-1, outputs.size(-1))
    targets = targets.view(-1)
    loss = nn.functional.cross_entropy(outputs, targets, ignore_index=-1)
    return loss

# This is a more robust version of the calc_loss_loader function
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches_processed = 0
    data_iterator = iter(data_loader)
    while True:
        if num_batches is not None and num_batches_processed >= num_batches:
            break
        try:
            input_batch, _ = next(data_iterator)
        except StopIteration:
            break
        input_batch = input_batch.to(device)
        with torch.no_grad():
            outputs = model(input_batch[:, :-1])
            loss = calc_loss_batch(outputs, input_batch[:, 1:])
            total_loss += loss.item()
            num_batches_processed += 1
    if num_batches_processed > 0:
        return total_loss / num_batches_processed
    else:
        return 0.0

# This is a more robust version of the evaluate_model function
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# This is a more robust version of the train_model_simple function
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                        eval_freq, eval_iter, start_context, tokenizer, checkpoint_path, 
                        grad_accum_steps=4, best_val_loss=float('inf'), scheduler=None):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    scaler = GradScaler()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for step, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            inputs = input_batch[:, :-1]
            targets = input_batch[:, 1:]
            with autocast():
                outputs = model(inputs)
                loss = calc_loss_batch(outputs, targets)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'epoch': epoch,
                        'global_step': global_step
                    }, checkpoint_path)
    return train_losses, val_losses, track_tokens_seen


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text))

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids)


# Note: The following imports are placeholders as the actual modules were not provided
try:
    from Instructional.model import GPTModel, CHOOSE_MODEL, BASE_CONFIG
    from Instructional.Data.data_set import InstructionDataset
    from Instructional.Data.collate import custom_collate_fn
    from Instructional.Data.format import format_input
    from Instructional.Accuracy.post_training import post_training_generate
    from Instructional.Training.generate_text import generate
    from GPT_Model.functions import text_to_token_ids, token_ids_to_text
except ImportError:
    print("Warning: Some modules could not be imported. Assuming they are available in your environment.")
    
# ---------------------- Backup Save Function ----------------------
def backup_save():
    print("\n❗ Detected interruption. Saving backup checkpoint...")
    # Get the current model's state dict and optimizer state
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    best_loss = best_val_loss
    
    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'best_val_loss': best_loss,
    }, stage1_checkpoint_path)
    print(f"✅ Backup saved to: {stage1_checkpoint_path}")

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
    model = GPTModel(BASE_CONFIG)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # You can adjust num_epochs_stage1 based on your GPU memory
    num_epochs_stage1 = 2
    best_val_loss = float('inf')

    try:
        train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs_stage1, eval_freq=50, eval_iter=5,
            start_context=format_input(val_data_json[0]), tokenizer=tokenizer,
            checkpoint_path=stage1_checkpoint_path,
            grad_accum_steps=4,
            best_val_loss=best_val_loss
        )
        print(f"✅ Stage 1 training completed. Final checkpoint saved to {stage1_checkpoint_path}.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❗ Stage 1 stopped due to out-of-memory error. Using latest checkpoint for Stage 2.")
        else:
            raise
    
    # ---------------------- Transition to Stage 2 ----------------------
    # If Stage 1 completed successfully, we will transition to Stage 2
    # seamlessly by falling through to the next block.
else:
    print(f"\n--- Stage 1 checkpoint found ({stage1_checkpoint_path}). Skipping to Stage 2 ---")
    
# ---------------------- STAGE 2: LoRA Fine-Tuning ----------------------
model = GPTModel(BASE_CONFIG)

try:
    model.load_state_dict(torch.load(stage1_checkpoint_path, map_location=device)['model_state_dict'])
    print(f"✅ Loaded Stage 1 checkpoint from {stage1_checkpoint_path} for LoRA fine-tuning.")
except FileNotFoundError:
    print(f"Error: Stage 1 checkpoint not found at {stage1_checkpoint_path}. Exiting.")
    exit()

# Apply LoRA layers to the loaded model
for block in model.trf_blocks:
    if isinstance(block.att.W_query, nn.Linear):
        block.att.W_query = LoRALinear(block.att.W_query.in_features, block.att.W_query.out_features)
    if isinstance(block.att.W_key, nn.Linear):
        block.att.W_key = LoRALinear(block.att.W_key.in_features, block.att.W_key.out_features)
    if isinstance(block.att.W_value, nn.Linear):
        block.att.W_value = LoRALinear(block.att.W_value.in_features, block.att.W_value.out_features)
    if isinstance(block.att.out_proj, nn.Linear):
        block.att.out_proj = LoRALinear(block.att.out_proj.in_features, block.att.out_proj.out_features)

# Freeze all original parameters, keeping only LoRA layers trainable
for name, param in model.named_parameters():
    if "A" not in name and "B" not in name:
        param.requires_grad = False

# Re-initialize the optimizer with ONLY the trainable parameters
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=1e-3, weight_decay=0.01)

# We can afford more epochs now due to less memory usage
num_epochs_stage2 = 20
best_val_loss = float('inf')

try:
    train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs_stage2, eval_freq=50, eval_iter=5,
        start_context=format_input(val_data_json[0]), tokenizer=tokenizer,
        checkpoint_path=final_checkpoint_path,
        grad_accum_steps=4,
        best_val_loss=best_val_loss
    )
    print(f"✅ Stage 2 training completed. Final checkpoint saved to {final_checkpoint_path}.")
except Exception as e:
    print(f"An error occurred during Stage 2 training: {e}")
    exit()

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

