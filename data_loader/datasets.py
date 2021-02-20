import torch
from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OCRDataset(Dataset):
    def __init__(self, image_dir, gt_path, label_dict, reshape_size, version='2015', transform=None):
        assert os.path.isdir(image_dir), f'dir \'{image_dir}\' not found!'
        self.image_dir = image_dir

        assert os.path.isfile(gt_path), f'file \'{gt_path}\' not found!'

        ch2ind, _ = get_label_dict(label_dict)

        self.labels = {}
        if version == '2015':
            with open(gt_path, 'r', encoding='utf-8-sig')as f:
                for line in f:
                    line = line.strip()
                    items = line.split(', ')
                    if len(items) != 2:
                        continue
                    image_name = items[0]
                    transcript = items[1][1:-1]

                    # encoded_trans = [ch2ind['SOS']]
                    encoded_trans = []
                    for ch in transcript:
                        if ch == ' ':
                            encoded_trans.append(ch2ind['SPACE'])
                        elif ch not in ch2ind.keys():
                            encoded_trans.append(ch2ind['UNK'])
                        else:
                            encoded_trans.append(ch2ind[ch])
                    # encoded_trans.append(ch2ind['EOS'])

                    self.labels[image_name] = encoded_trans

        elif version == 'Synth90k':
            with open(gt_path, 'r')as f:
                print('Loading label file...')
                for line in tqdm(f):
                    line = line.strip()
                    image_name = line.split(' ')[0]
                    transcript = image_name.split('_')[1]

                    # encoded_trans = [ch2ind['SOS']]
                    encoded_trans = []
                    for ch in transcript:
                        if ch == ' ':
                            encoded_trans.append(ch2ind['SPACE'])
                        elif ch not in ch2ind.keys():
                            encoded_trans.append(ch2ind['UNK'])
                        else:
                            encoded_trans.append(ch2ind[ch])
                    # encoded_trans.append(ch2ind['EOS'])

                    self.labels[image_name] = encoded_trans

        self.transform = transform
        self.tgt_width = reshape_size[0]
        self.tgt_height = reshape_size[1]

    def __getitem__(self, index):
        label = list(self.labels.values())[index]
        image_name = list(self.labels.keys())[index]

        # process image
        image_path = os.path.join(self.image_dir, image_name)
        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size
        reshape_width = self.tgt_height * (width / height)
        img = img.resize([int(reshape_width), int(self.tgt_height)])
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


def get_label_dict(label_dict):
    chs = []
    with open(label_dict, 'r')as f:
        for line in f:
            line = line.strip()
            ch = line.split(' ')[1]
            chs.append(ch)
    ch2ind = {}
    ind2ch = {}
    for i in range(len(chs)):
        ch2ind[chs[i]] = i
        ind2ch[i] = chs[i]
    return ch2ind, ind2ch
