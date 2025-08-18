import torch
import tiktoken

from Classification.model import model
from Classification.Data.data_loaders import train_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = tiktoken.get_encoding("gpt2")


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :] # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

# --- Test samples ---
model_state_dict = torch.load("review_classifier.pth")
model.load_state_dict(model_state_dict)


text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))  # Expected: HAM


text_4 = (
    "Don’t forget to bring your laptop for tomorrow’s project meeting."
)
print(classify_review(
    text_4, model, tokenizer, device, max_length=train_dataset.max_length
))  # Expected: HAM


text_6 = (
    "Happy birthday! 🎉 Wishing you an amazing year ahead."
)
print(classify_review(
    text_6, model, tokenizer, device, max_length=train_dataset.max_length
))  # Expected: HAM

text_7 = (
    "Limited time offer!!! Buy one get one free on all electronics."
)
print(classify_review(
    text_7, model, tokenizer, device, max_length=train_dataset.max_length
))  # Expected: SPAM

text_8 = (
    "I’ll be late to the call, please start without me."
)
print(classify_review(
    text_8, model, tokenizer, device, max_length=train_dataset.max_length
))  # Expected: HAM

