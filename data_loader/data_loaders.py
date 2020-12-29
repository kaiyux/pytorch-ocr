import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from base import BaseDataLoader
from .datasets import *


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ICDARDataLoader(BaseDataLoader):
    def __init__(self, image_dir, gt_path, label_dict, version, batch_size, reshape_size,
                 shuffle=True, validation_split=0.0, num_workers=1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = ICDARDataset(image_dir, gt_path, label_dict, reshape_size, version, transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         collate_fn=icdar_collate_fn)


def icdar_collate_fn(batch):
    max_len = 0
    for p in batch:
        max_len = max(max_len, len(p[1]))
    for i in range(len(batch)):
        while len(batch[i][1]) < max_len:
            batch[i][1].append(2)
        batch[i] = (batch[i][0], torch.LongTensor(batch[i][1]))
    return default_collate(batch)
