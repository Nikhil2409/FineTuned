import torch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches_processed = 0
    # Use an iterator to handle an unknown number of batches
    data_iterator = iter(data_loader)

    while True:
        # Stop the loop if we've processed the desired number of batches
        if num_batches is not None and num_batches_processed >= num_batches:
            break
        
        try:
            # Get the next batch from the iterator
            input_batch, target_batch = next(data_iterator)
        except StopIteration:
            # Break the loop if the data loader is exhausted
            break
        
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        with torch.no_grad():
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            num_batches_processed += 1

    if num_batches_processed > 0:
        return total_loss / num_batches_processed
    else:
        # Return 0 or handle the case where no batches were processed
        return 0.0