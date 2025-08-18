import torch
import os
from Classification.Accuracy.accuracy import calc_accuracy_loader
from Classification.model import model
from Classification.Data.data_loaders import test_loader, train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
checkpoint_path = "Classification/review_classifier.pth"

if os.path.exists(checkpoint_path):
    print(f"[INFO] Loading model from {checkpoint_path}")
    model_state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
else:
    print(f"[INFO] No checkpoint found at {checkpoint_path}, starting from scratch")
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
