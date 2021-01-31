import os
from PIL import Image

image_dir = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT'
gt_path = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/Challenge2_Training_Task3_Images_GT/gt.txt'

patched_image_dir = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/patched'
patched_gt_path = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/patched/gt.txt'

tgt_width = 3200
tgt_height = 64

patched_width = 0
image_idx = 0
tgt_image = Image.new('RGB', (tgt_width, tgt_height))
transcripts = []
cur_trans = []
with open(gt_path, 'r', encoding='utf-8-sig')as f:
    for line in f:
        line = line.strip()
        items = line.split(', ')
        if len(items) != 2:
            continue
        image_name = items[0]
        transcript = items[1][1:-1]

        image_path = os.path.join(image_dir, image_name)
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        reshape_width = int(tgt_height * (width / height))

        if patched_width + reshape_width > tgt_width:
            tgt_image.save(os.path.join(patched_image_dir, str(image_idx) + '.jpg'), quality=100)
            tgt_image.close()
            patched_width = 0
            image_idx += 1
            tgt_image = Image.new('RGB', (tgt_width, tgt_height))
            transcripts.append(cur_trans)
            cur_trans = []

        img = img.resize([reshape_width, tgt_height])

        tgt_image.paste(img, (patched_width, 0, patched_width + reshape_width, tgt_height))
        patched_width += reshape_width
        cur_trans.append(transcript)

        img.close()

tgt_image.save(os.path.join(patched_image_dir, str(image_idx) + '.jpg'), quality=100)
tgt_image.close()
transcripts.append(cur_trans)

with open(patched_gt_path, 'w')as f:
    for i, trans in enumerate(transcripts):
        line = str(i) + '.jpg, "' + ''.join(trans)
        f.write(line + '"\n')
