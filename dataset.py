import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

class NewsDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        feats = self.dataset[index][0]
        label = self.dataset[index][1]
        return torch.tensor(feats), torch.tensor(label)
    
    def __len__(self):
        return len(self.dataset)


