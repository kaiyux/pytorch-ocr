import os
from PIL import Image
from tqdm import tqdm

image_dir = '/home/stu7/workspace/ocr/dataset/recog/images'
gt_path = '/home/stu7/workspace/ocr/dataset/recog/gt.txt'

patched_image_dir = '/home/stu7/workspace/ocr/dataset/recog/patched/images'
patched_gt_path = '/home/stu7/workspace/ocr/dataset/recog/patched/gt.txt'

tgt_width = 1600
tgt_height = 64
dataset_type = '2015'

patched_width = 0
image_idx = 0
tgt_image = Image.new('RGB', (tgt_width, tgt_height))
transcripts = []
cur_trans = []

num_small = 0
num_wide = 0

with open(gt_path, 'r', encoding='utf-8-sig')as f:
    for line in tqdm(f):
        line = line.strip()
        if dataset_type == '2015':
            items = line.split(', ')
            if len(items) != 2:
                continue
            image_name = items[0]
            transcript = items[1][1:-1]
        else:
            image_name = line.split(' ')[0]
            transcript = image_name.split('_')[1]

        image_path = os.path.join(image_dir, image_name)
        try:
            img = Image.open(image_path).convert("RGB")
        except:
            continue

        width, height = img.size
        if max(width, height) < 30 or width < height:
            num_small += 1
            continue

        reshape_width = int(tgt_height * (width / height))
        if reshape_width > tgt_width:
            num_wide += 1
            continue

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
print(f'Done. Total nums after patch:{image_idx + 1}')
print(f'{num_small} images skipped because of too small.')
print(f'{num_wide} images skipped because of too wide.')
