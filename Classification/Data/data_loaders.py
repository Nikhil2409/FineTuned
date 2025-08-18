from torch.utils.data import DataLoader
import torch
import tiktoken
import os

from Classification.Data.data_set import SpamDataset

# ----------------------------
# Config
# ----------------------------
num_workers = 0
batch_size = 8
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)

data_dir = os.path.join("Classification", "Data") 

train_dataset = SpamDataset(
    csv_file=os.path.join(data_dir, "train.csv"),
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file=os.path.join(data_dir, "validation.csv"),
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file=os.path.join(data_dir, "test.csv"),
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
