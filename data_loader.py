import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class CIFAR100Dataset(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        self.dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=True,
            transform=transform if transform else self.default_transform()
        )
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize(32),  # StyleGAN2 입력 크기에 맞춤
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

def get_dataloader(batch_size=64, num_workers=4, train=True):
    dataset = CIFAR100Dataset(train=train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader