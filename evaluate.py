import torch
from torch import nn, optim
from tqdm import tqdm

def evaluate(model, data_loader, criterion, device="cuda:0"):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        data_tqdm = tqdm(data_loader, leave=True)
        for idx, batch in enumerate(data_tqdm):
            inputs, seq_len, labels = batch
            inputs, seq_len, labels = inputs.to(device), seq_len.int(), labels.to(device)
            logits = model(inputs, seq_len)
            loss= criterion(logits, labels)
            total_loss += loss.item()
            _, predictions = torch.max(logits, 1)
            total += labels.size(0)
            total_correct += (predictions==labels).sum().item()
            data_tqdm.set_postfix(loss=loss.item(), accuracy=(predictions==labels).sum().item())
    accuracy = total_correct/total
    return accuracy, total_loss