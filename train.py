import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np

def train(model, data_loader, optimizer, criterion, total_epochs, save_path, device="cuda:0"):
    model.train()
    min_loss = np.inf
    for epoch in range(total_epochs):
        data_tqdm = tqdm(data_loader, leave=True)
        total_loss = 0.0
        for idx, batch in enumerate(data_tqdm):
            inputs, labels = batch
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss < min_loss:
            min_loss = total_loss
            state = {"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "min_loss": min_loss
                    }
            torch.save(state, save_path)