import torch
import torch.nn as nn

def calc_loss_batch(outputs, targets):
    outputs = outputs.reshape(-1, outputs.size(-1))
    targets = targets.reshape(-1)
    loss = torch.nn.functional.cross_entropy(outputs, targets, ignore_index=-1)
    return loss


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
            inputs = input_batch[:, :-1].squeeze(1)
            targets = input_batch[:, 1:].squeeze()

            outputs = model(inputs)
        
            loss = calc_loss_batch(outputs, targets)
            
            total_loss += loss.item()
            num_batches_processed += 1

    if num_batches_processed > 0:
        return total_loss / num_batches_processed
    else:
        return 0.0