import json
from datasets import load_dataset

# 🔹 Load the dataset
data = load_dataset("Tural/stanford_alpaca")
full_dataset = data['train']

# 🔹 Split: 85% train, 15% test → then split test into test+val
train_test_split = full_dataset.train_test_split(test_size=0.15, seed=42)
test_val_split = train_test_split['test'].train_test_split(test_size=0.33, seed=42)

# Assign splits
train_data = train_test_split['train']
test_data = test_val_split['train']
val_data = test_val_split['test']

# 🔹 Save JSONs
with open("train_data.json", "w") as f:
    json.dump(train_data.to_list(), f)

with open("val_data.json", "w") as f:
    json.dump(val_data.to_list(), f)

with open("test_data.json", "w") as f:
    json.dump(test_data.to_list(), f)

# 🔹 Print sizes
print(f"Total dataset size: {len(full_dataset)}")
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Testing data size: {len(test_data)}")
