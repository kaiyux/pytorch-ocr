from PIL import Image
import json
import os
import re
import shutil
import scipy.io as sio
from tqdm import tqdm
from utils.util import contain_chinese


class CustomDataset(object):
    def __init__(self, image_dir, gt_file):
        print('Initialize dataset...')
        self.image_dir = image_dir
        self.gt_file = gt_file
        self.labels = {}

    def crop(self):
        pass


class ICDAR2013(CustomDataset):
    def __init__(self, image_dir, gt_file):
        super().__init__(image_dir, gt_file)
        with open(gt_file, 'r', encoding='utf-8-sig')as f:
            for line in tqdm(f):
                line = line.strip()
                items = line.split(', ')
                if len(items) != 2:
                    continue
                image_name = items[0]
                transcript = items[1][1:-1]

                self.labels[os.path.join(image_dir, image_name)] = transcript


class ICDAR2017(CustomDataset):
    def __init__(self, image_dir, gt_file):
        super().__init__(image_dir, gt_file)
        with open(gt_file, 'r', encoding='utf-8-sig')as f:
            for line in tqdm(f):
                line = line.strip()
                items = line.split(',')
                if len(items) != 3:
                    continue
                image_name = items[0]
                text_type = items[1]
                if text_type != 'Latin':
                    continue
                transcript = items[2]

                self.labels[os.path.join(image_dir, image_name)] = transcript


class ICDAR2019(CustomDataset):
    def __init__(self, image_dir, gt_file, crop_dir):
        super().__init__(image_dir, gt_file)
        self.crop_dir = crop_dir
        with open(gt_file, 'r') as f:
            self.icdar = json.load(f)
        self.crop()

    def crop(self):
        index = 0
        print('Crop image...')
        for image_name in tqdm(self.icdar.keys()):
            image_path = os.path.join(self.image_dir, image_name + '.jpg')
            try:
                img = Image.open(image_path)
            except:
                continue
            for ann in self.icdar[image_name]:
                if not ann["illegibility"]:
                    transcription = ann['transcription']
                    if contain_chinese(transcription) or ('language' in ann.keys() and ann['language'] != 'Latin'):
                        continue
                    xmin = float('inf')
                    xmax = 0
                    ymin = float('inf')
                    ymax = 0
                    for point in ann["points"]:
                        xmin = min(xmin, point[0])
                        xmax = max(xmax, point[0])
                        ymin = min(ymin, point[1])
                        ymax = max(ymax, point[1])

                    bbox = [xmin, ymin, xmax, ymax]

                    text_region = img.crop(bbox)
                    region_path = os.path.join(self.crop_dir, 'img_' + str(index) + '.jpg')
                    text_region.save(region_path)
                    self.labels[region_path] = transcription
                    index += 1
            img.close()


class COCOText(CustomDataset):
    def __init__(self, image_dir, gt_file):
        super().__init__(image_dir, gt_file)
        with open(gt_file, 'r', encoding='utf-8')as f:
            for line in tqdm(f):
                line = line.strip()
                items = line.split(',')
                if len(items) != 2:
                    continue
                image_name = items[0] + '.jpg'
                transcript = items[1]

                self.labels[os.path.join(image_dir, image_name)] = transcript


class Synth90k(CustomDataset):
    def __init__(self, image_dir, gt_file, nums=None):
        super().__init__(image_dir, gt_file)
        index = 0
        with open(gt_file, 'r')as f:
            for line in tqdm(f):
                if nums is not None and nums == index:
                    break
                line = line.strip()
                image_name = line.split(' ')[0]
                transcript = image_name.split('_')[1]

                image_path = os.path.join(image_dir, image_name)
                try:
                    img = Image.open(image_path)
                    w = img.size[0]
                    h = img.size[1]
                    if w < h or w * h < 100 or w * h > 1000000:
                        continue
                    img.close()
                except:
                    continue

                self.labels[image_path] = transcript
                index += 1


class SynthText(CustomDataset):
    def __init__(self, image_dir, gt_file, crop_dir, nums=None):
        super().__init__(image_dir, gt_file)
        self.crop_dir = crop_dir
        self.nums = nums

        self.synth_data = sio.loadmat(gt_file)
        self.imnames = self.synth_data['imnames'][0]
        self.coords = self.synth_data["wordBB"][0]
        self.transcription = self.synth_data["txt"][0]

        self.crop()

    def crop(self):
        index = 0
        print('Crop image...')
        for i in tqdm(range(len(self.imnames))):
            image_name = self.imnames[i][0]
            image_path = os.path.join(self.image_dir, image_name)
            try:
                img = Image.open(image_path)
                w = img.size[0]
                h = img.size[1]
                if w < h or w * h < 100 or w * h > 1000000:
                    img.close()
                    continue
            except:
                continue
            raw_texts = self.transcription[i]
            texts = []
            for t in raw_texts:
                texts.extend(re.split(r'[\s\n]', t.strip()))
            while '' in texts:
                texts.remove('')
            coord = self.coords[i]
            if len(coord.shape) != 3 or (len(coord.shape) == 3 and len(texts) != coord.shape[2]):
                continue
            for j in range(len(coord[0][0])):
                x1 = coord[0][0][j]
                y1 = coord[1][0][j]
                x2 = coord[0][1][j]
                y2 = coord[1][1][j]
                x3 = coord[0][2][j]
                y3 = coord[1][2][j]
                x4 = coord[0][3][j]
                y4 = coord[1][3][j]
                xmin = min(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                xmax = max(x1, x2, x3, x4)
                ymax = max(y1, y2, y3, y4)

                bbox = [xmin, ymin, xmax, ymax]
                trans = texts[j]

                try:
                    text_region = img.crop(bbox)
                    region_path = os.path.join(self.crop_dir, 'img_' + str(index) + '.jpg')
                    text_region.save(region_path)
                    self.labels[region_path] = trans
                    index += 1
                except:
                    continue
                if self.nums is not None and self.nums == index:
                    return
            img.close()


if __name__ == '__main__':
    icdar2013 = ICDAR2013(
        '/home/xiekaiyu/ocr/dataset/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT',
        '/home/xiekaiyu/ocr/dataset/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT/gt.txt')
    icdar2015 = ICDAR2013(
        '/home/xiekaiyu/ocr/dataset/ICDAR2015WordRecognition/ch4_training_word_images_gt',
        '/home/xiekaiyu/ocr/dataset/ICDAR2015WordRecognition/ch4_training_word_images_gt/gt.txt')
    icdar2017 = ICDAR2017(
        '/home/xiekaiyu/ocr/dataset/ICDAR2017MLTRec/train',
        '/home/xiekaiyu/ocr/dataset/ICDAR2017MLTRec/train_gt/gt.txt')
    icdar2019_lsvt = ICDAR2019(
        '/home/xiekaiyu/ocr/dataset/ICDAR2019LSVT/train_full_images',
        '/home/xiekaiyu/ocr/dataset/ICDAR2019LSVT/train_full_labels.json',
        '/home/xiekaiyu/ocr/dataset/ICDAR2019LSVT/crop')
    icdar2019_art = ICDAR2019(
        '/home/xiekaiyu/ocr/dataset/ICDAR2019ArT/train_images',
        '/home/xiekaiyu/ocr/dataset/ICDAR2019ArT/train_labels.json',
        '/home/xiekaiyu/ocr/dataset/ICDAR2019ArT/crop')
    cocotext = COCOText(
        '/home/xiekaiyu/ocr/dataset/COCO-Text-words-trainval/train_words',
        '/home/xiekaiyu/ocr/dataset/COCO-Text-words-trainval/train_words_gt.txt')
    synth90k = Synth90k(
        '/home/xiekaiyu/ocr/dataset/Synth90k/90kDICT32px',
        '/home/xiekaiyu/ocr/dataset/Synth90k/90kDICT32px/annotation_train.txt',
        nums=160000)
    synth_text = SynthText(
        '/home/xiekaiyu/ocr/dataset/SynthText/SynthText',
        '/home/xiekaiyu/ocr/dataset/SynthText/SynthText/gt.mat',
        '/home/xiekaiyu/ocr/dataset/SynthText/crop',
        nums=160000)
    datasets = [icdar2013, icdar2015, icdar2017, icdar2019_lsvt, icdar2019_art, cocotext, synth90k, synth_text]
    output_dir = '/home/xiekaiyu/workspace/pytorch-ocr/data/images'
    gt_file = '/home/xiekaiyu/workspace/pytorch-ocr/data/gt.txt'
    index = 0
    with open(gt_file, 'w', encoding='utf-8')as f:
        for dataset in datasets:
            labels = dataset.labels
            for img_path in tqdm(labels.keys()):
                # copy image
                image_name = img_path.split('/')[-1]
                img_type = image_name.split('.')[-1]
                image_name_new = 'img_' + str(index) + '.' + img_type
                destination = os.path.join(output_dir, image_name_new)
                shutil.copy(img_path, destination)

                # write gt file
                line = image_name_new + ', "' + labels[img_path] + '"\n'
                f.write(line)

                index += 1
    print('Dataset generated. Including {} images.'.format(index))
