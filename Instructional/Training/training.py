import os, re, time, torch, tiktoken
from Instructional.model import model, CHOOSE_MODEL, BASE_CONFIG
from Instructional.Training.functions import train_model_simple
from Instructional.Data.format import format_input
from Instructional.Data.data_loader import train_loader, val_loader, val_data, test_data
from Instructional.Accuracy.post_training import post_training_generate

start_time = time.time()
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Save to Google Drive so it persists
base_dir = "/content/drive/MyDrive/Finetuned_checkpoints"
os.makedirs(base_dir, exist_ok=True)

checkpoint_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
checkpoint_path = os.path.join(base_dir, checkpoint_name)

# Load checkpoint if available
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")
except FileNotFoundError:
    print("No checkpoint found, training from scratch.")

model = model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)

num_epochs = 1
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer,
    grad_accum_steps=4
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60

torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved as {checkpoint_path}")
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 🔹 Save responses also in Drive
output_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-responses.json"
output_path = os.path.join(base_dir, output_name)
test_data = post_training_generate(model, tokenizer, device, test_data)

print(f"Responses saved at {output_path}")
