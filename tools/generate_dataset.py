from PIL import Image
import json
import os
import shutil
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


if __name__ == '__main__':
    icdar2013 = ICDAR2013(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT/gt.txt')
    icdar2015 = ICDAR2013(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2015WordRecognition/ch4_training_word_images_gt',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2015WordRecognition/ch4_training_word_images_gt/gt.txt')
    icdar2017 = ICDAR2017(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2017MLTRec/train',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2017MLTRec/train_gt/gt.txt')
    icdar2019_lsvt = ICDAR2019(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2019LSVT/train_full_images',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2019LSVT/train_full_labels.json',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2019LSVT/crop')
    icdar2019_art = ICDAR2019(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2019ArT/train_images',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2019ArT/train_labels.json',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2019ArT/crop')
    datasets = [icdar2013, icdar2015, icdar2017, icdar2019_lsvt, icdar2019_art]
    output_dir = '/home/stu7/workspace/ocr/dataset/recog/images'
    gt_file = '/home/stu7/workspace/ocr/dataset/recog/gt.txt'
    index = 0
    with open(gt_file, 'w')as f:
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
