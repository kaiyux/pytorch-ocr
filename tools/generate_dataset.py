import os
import shutil
from tqdm import tqdm


class CustomDataset(object):
    def __init__(self, image_dir, gt_file):
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


if __name__ == '__main__':
    icdar2013 = ICDAR2013(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT/gt.txt')
    icdar2015 = ICDAR2013(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2015WordRecognition/ch4_training_word_images_gt',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2015WordRecognition/ch4_training_word_images_gt/gt.txt')
    icdar2017 = ICDAR2017(
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2017MLTRec/train',
        '/home/stu7/workspace/ocr/dataset/all/ICDAR2017MLTRec/train_gt/gt.txt'
    )
    datasets = [icdar2013, icdar2015, icdar2017]

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
