import time
import os
import torch
from Classification.model import model
from Classification.Training.functions import train_classifier_simple
from Classification.Data.data_loaders import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_time = time.time()

torch.manual_seed(123)

checkpoint_path = "Classification/review_classifier.pth"
start_epoch = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

if os.path.exists(checkpoint_path):
    print(f"[INFO] Loading existing model state from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("[INFO] Model state loaded. Training will resume from scratch (epochs not tracked in this setup).")
else:
    print("[INFO] No existing model state found. Starting training from scratch.")


num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
torch.save(model.state_dict(), "Classification/review_classifier.pth")
print(f"Training completed in {execution_time_minutes:.2f} minutes.")