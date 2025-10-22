import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.cuda # Needed for GPU memory logging
# Assuming these are accessible via the Python path
from GPT_Model.functions import text_to_token_ids, token_ids_to_text
from Instructional.Training.generate_text import generate
from Instructional.Training.loss import calc_loss_batch, calc_loss_loader


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, checkpoint_path, 
                       grad_accum_steps=4, best_val_loss=float('inf'), scheduler=None, writer=None): # <-- CORRECTED SIGNATURE
    
    current_best_val_loss = best_val_loss 
    global_step = 0
    tokens_seen = 0
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
                loss = calc_loss_batch(outputs, targets,model,device)
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
            
            # --- Periodic Evaluation and Checkpointing ---
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                
                # --- TENSORBOARD LOGGING (Learning Curve) ---
                if writer is not None:
                    # Log Loss and Learning Rate
                    writer.add_scalar('Loss/Train', train_loss, global_step)
                    writer.add_scalar('Loss/Validation', val_loss, global_step)
                    writer.add_scalar('Metrics/Tokens_Seen', tokens_seen, global_step)
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Optimizer/Learning_Rate', current_lr, global_step)
                    
                    # Log GPU Memory Usage (for efficiency defense)
                    if device.type == 'cuda':
                        max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                        writer.add_scalar('System/GPU_Memory_Max_GB', max_mem_gb, global_step)

                # --- Console Print (Keep this for immediate visibility) ---
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                # --- Checkpoint Saving ---
                if val_loss < current_best_val_loss:
                    current_best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': current_best_val_loss,
                        'epoch': epoch,
                        'global_step': global_step
                    }, checkpoint_path)
                    
    # Return only the final best loss (as requested for compatibility)
    return None, None, current_best_val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    # Assuming text_to_token_ids, generate, and token_ids_to_text functions are available
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