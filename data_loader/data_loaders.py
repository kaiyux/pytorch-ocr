import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from base import BaseDataLoader
from .datasets import *


class OCRDataLoader(BaseDataLoader):
    def __init__(self, image_dir, gt_path, label_dict, version, batch_size, reshape_size,
                 shuffle=True, validation_split=0.0, num_workers=1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.dataset = OCRDataset(image_dir, gt_path, label_dict, reshape_size, version, transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         collate_fn=icdar_collate_fn)


def icdar_collate_fn(batch):
    # patch every `patch_num` images
    patch_num = 8
    batch_size = len(batch)

    if batch_size > patch_num and batch_size % patch_num == 0:
        batch = sorted(batch, key=lambda img_label_pair: img_label_pair[0].shape[2])

        new_batch = []
        i = 0
        while i < int(batch_size / 2):
            imgs = []
            new_label = []
            for j in range(patch_num // 2):
                imgs.append(batch[i + j][0])
                new_label.extend(batch[i + j][1])
                imgs.append(batch[-(i + j) - 1][0])
                new_label.extend(batch[-(i + j) - 1][1])
            i += patch_num // 2
            new_img = torch.cat(tuple(imgs), dim=2)
            new_label.append(2)  # EOS
            new_batch.append((new_img, new_label))
        batch = new_batch
    else:
        for p in batch:
            p[1].append(2)  # EOS

    max_width = 0
    max_len = 0
    for p in batch:
        max_width = max(max_width, p[0].shape[2])  # image
        max_len = max(max_len, len(p[1]))  # target sequence

    for i in range(len(batch)):
        if batch[i][0].shape[2] < max_width:
            pad_width = max_width - batch[i][0].shape[2]
            batch[i] = (torch.nn.functional.pad(batch[i][0], (0, pad_width,), "constant", 0), batch[i][1])
        while len(batch[i][1]) < max_len:
            batch[i][1].append(0)
        batch[i] = (batch[i][0], torch.LongTensor(batch[i][1]))

    return default_collate(batch)
