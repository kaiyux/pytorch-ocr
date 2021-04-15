import lmdb
import os
import io
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


def get_image(image_path, tgt_height=64, tgt_width=320):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    reshape_width = int(tgt_height * (width / height))
    if reshape_width >= tgt_width:
        resize = transforms.Resize([tgt_height, tgt_width])
        img = resize(img)
    else:
        resize = transforms.Resize([tgt_height, reshape_width])
        img = resize(img)
        pad_width = tgt_width - reshape_width
        pad = transforms.Pad(padding=(0, 0, int(pad_width), 0))
        img = pad(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img = transform(img)

    return img


def main():
    images_dir = '/home/xiekaiyu/workspace/pytorch-ocr/synth_data/images'
    gt_file = '/home/xiekaiyu/workspace/pytorch-ocr/synth_data/gt.txt'
    output = '/home/xiekaiyu/workspace/pytorch-ocr/synth_data/synth_data.lmdb'
    assert not os.path.exists(output)
    env = lmdb.open(output, map_size=1099511627776)
    txn = env.begin(write=True)
    commit_every_img = 100
    print(f'Start writing to {output}...')
    with open(gt_file, 'r', encoding='utf-8')as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip()
            items = line.split(', ')
            if len(items) != 2:
                continue
            image_name = items[0]
            image_path = os.path.join(images_dir, image_name)

            image_tensor = get_image(image_path)
            buff = io.BytesIO()
            torch.save(image_tensor, buff)
            buff.seek(0)

            txn.put(key=image_name.encode(), value=buff.read())
            del buff
            if i % commit_every_img == 0:
                txn.commit()
                txn = env.begin(write=True)

    env.close()
    print('Done.')


if __name__ == '__main__':
    main()
