from PIL import Image
import os

image_dir = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT'
gt_file = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT/gt.txt'
output_file = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT/sorted_gt.txt'

name2width = {}
name2trans = {}
with open(gt_file, 'r', encoding='utf-8-sig')as f:
    for line in f:
        line = line.strip()
        items = line.split(', ')
        if len(items) != 2:
            continue
        image_name = items[0]
        transcript = items[1][1:-1]
        name2trans[image_name] = transcript

        image_path = os.path.join(image_dir, image_name)
        img = Image.open(image_path).convert("RGB")
        tgt_height = 64
        width, height = img.size
        reshape_width = tgt_height * (width / height)
        name2width[image_name] = reshape_width
        img.close()

sorted_name2width = sorted(name2width.items(), key=lambda x: x[1], reverse=False)
names = [item[0] for item in sorted_name2width]

with open(output_file, 'w')as o:
    for name in names:
        o.write(name + ', "' + name2trans[name] + '"\n')
