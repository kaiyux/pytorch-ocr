from base import BaseDataLoader
from .datasets import *


class OCRDataLoader(BaseDataLoader):
    def __init__(self, image_dir, gt_path, label_dict, version, batch_size, reshape_size,
                 shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = OCRDataset(image_dir, gt_path, label_dict, reshape_size, version)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         collate_fn=icdar_collate_fn)


def icdar_collate_fn(batch):
    images = []
    labels = []
    for p in batch:
        images.append(p[0])
        labels.extend(p[1])
    images = torch.stack(images, dim=0)
    return images, labels
