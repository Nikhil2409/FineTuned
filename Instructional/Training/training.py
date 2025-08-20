import os, re, time, math, torch, tiktoken, json
from functools import partial
from torch.utils.data import DataLoader
import torch.nn as nn

from Instructional.Training.functions import train_model_simple
from Instructional.model import GPTModel, CHOOSE_MODEL, BASE_CONFIG
from Instructional.Data.data_set import InstructionDataset
from Instructional.Data.collate import custom_collate_fn
from Instructional.Data.format import format_input
from Instructional.Accuracy.post_training import post_training_generate

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

        # Frozen weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        # Trainable LoRA matrices
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scaling = self.alpha / self.r

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias) + \
               self.scaling * nn.functional.linear(x, self.B @ self.A)

# ---------------------- Initialize Fresh Model ----------------------
model = GPTModel(BASE_CONFIG)

# ---------------------- LoRA Replacement ----------------------
for block in model.trf_blocks:
    if isinstance(block.att.W_query, nn.Linear):
        block.att.W_query = LoRALinear(block.att.W_query.in_features, block.att.W_query.out_features)
    if isinstance(block.att.W_key, nn.Linear):
        block.att.W_key = LoRALinear(block.att.W_key.in_features, block.att.W_key.out_features)
    if isinstance(block.att.W_value, nn.Linear):
        block.att.W_value = LoRALinear(block.att.W_value.in_features, block.att.W_value.out_features)
    if isinstance(block.att.out_proj, nn.Linear):
        block.att.out_proj = LoRALinear(block.att.out_proj.in_features, block.att.out_proj.out_features)

# ---------------------- Freeze Non-LoRA Params ----------------------
for name, param in model.named_parameters():
    if "A" not in name and "B" not in name:
        param.requires_grad = False

# ---------------------- Load Dataset ----------------------
with open("train_data.json", "r") as f:
    train_data_json = json.load(f)
with open("val_data.json", "r") as f:
    val_data_json = json.load(f)
with open("test_data.json", "r") as f:
    test_data_json = json.load(f)

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

# ---------------------- Checkpoint ----------------------
base_dir = "/content/drive/MyDrive/Finetuned_checkpoints"
os.makedirs(base_dir, exist_ok=True)
checkpoint_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft-lora.pth"
checkpoint_path = os.path.join(base_dir, checkpoint_name)

# ---------------------- Optimizer ----------------------
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=5e-4, weight_decay=0.01)

# ---------------------- Load Checkpoint ----------------------
best_val_loss = float('inf')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    print(f"✅ Loaded checkpoint from {checkpoint_path} with best val loss: {best_val_loss:.3f}")
else:
    print("No checkpoint found, training from scratch.")

# ---------------------- Move model to device ----------------------
model = model.to(device)

# ---------------------- Train ----------------------
start_time = time.time()
num_epochs = 5
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data_json[0]), tokenizer=tokenizer,
    checkpoint_path=checkpoint_path,
    grad_accum_steps=4,
    best_val_loss=best_val_loss
)
end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")

# ---------------------- Post-Training Generation ----------------------
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded best checkpoint for generation.")
else:
    print("⚠️ No checkpoint found.")

output_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-responses.json"
output_path = os.path.join(base_dir, output_name)
test_data_json = post_training_generate(model, tokenizer, device, test_data_json)

with open(output_path, "w") as f:
    json.dump(test_data_json, f, indent=4)
print(f"Responses saved at {output_path}")
