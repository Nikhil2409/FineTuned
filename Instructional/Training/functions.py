from Instructional.Training.loss import calc_loss_batch, calc_loss_loader
import torch
from Instructional.Training.generate_text import generate
from GPT_Model.functions import text_to_token_ids, token_ids_to_text
from torch.cuda.amp import autocast, GradScaler

import os
import torch
from torch.cuda.amp import autocast, GradScaler

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, checkpoint_path, 
                       grad_accum_steps=4):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    scaler = GradScaler()

    # 🔹 New: Initialize a variable to track the best validation loss
    best_val_loss = float('inf')
    
    # 🔹 The path to save the best model checkpoint
    # Assumes checkpoint_path is passed in or defined in the calling scope
    # For your training.py, this would be:
    # checkpoint_path = f"/content/drive/MyDrive/Finetuned_checkpoints/{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for step, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with autocast():
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                # 🔹 New: Check for validation loss improvement and save the model
                if val_loss < best_val_loss:
                    print(f"Validation loss improved from {best_val_loss:.3f} to {val_loss:.3f}. Saving checkpoint.")
                    best_val_loss = val_loss
                    # Assuming `checkpoint_path` is available in the outer scope
                    torch.save(model.state_dict(), checkpoint_path)

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
# -----------------------------
# Helper functions remain the same
# -----------------------------
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
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
