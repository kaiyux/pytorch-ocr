import argparse
import torch
import model.model as module_arch
from parse_config import ConfigParser
import os
from tqdm import tqdm
import json
from utils.util import recognize


def prepare(args, resume):
    config = ConfigParser.from_args(args)
    logger = config.get_logger('prepare model')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, device


def main():
    image_path = '/home/stu7/workspace/ocr/dataset/all/ICDAR2019ArT/test_task2_images'
    resume = '/home/stu7/workspace/ocr/pytorch-ocr/recog_model/models/OCR/pretrained/checkpoint-epoch1.pth'
    label_dict = '/home/stu7/workspace/ocr/pytorch-ocr/label_dicts/label_dict_en.txt'
    output = '/home/stu7/workspace/ocr/pytorch-ocr/recog_model/models/OCR/pretrained/submit.json'
    model, device = prepare(args, resume)

    image_names = os.listdir(image_path)
    preds = []
    for img_name in tqdm(image_names):
        pred = recognize(os.path.join(image_path, img_name), model, label_dict, device)
        preds.append(pred)
    icdar2019art(preds, image_names, output)


def icdar2019art(preds, image_names, output):
    results = {}
    for i in range(len(preds)):
        img_name = 'res_' + image_names[i].split('_')[1].split('.')[0]
        results[img_name] = [{'transcription': preds[i]}]
    with open(output, 'w')as f:
        json.dump(results, f)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='evaluate')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    main()
