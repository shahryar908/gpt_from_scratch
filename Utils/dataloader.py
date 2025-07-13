import torch
from tokenizer import encoded
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, data, block_size,stride=None):
        self.data = data
        self.block_size = block_size
        self.stride=stride
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

block_size = 128
batch_size = 64

dataset = CharDataset(encoded, block_size,stride=128)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

