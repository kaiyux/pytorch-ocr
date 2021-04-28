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
    resume = '/home/xiekaiyu/workspace/pytorch-ocr/RecognizeModel/models/OCR/0331_033837/checkpoint-epoch43.pth'
    label_dict = '/home/xiekaiyu/workspace/pytorch-ocr/label_dicts/label_dict_en.txt'
    model, device = prepare(args, resume)

    icdar2013_image_path = '/home/xiekaiyu/ocr/dataset/ICDAR2013WordRecognition/Challenge2_Test_Task3_Images'
    icdar2013_output = '/home/xiekaiyu/ocr/dataset/ICDAR2013WordRecognition/eval/script_test_ch2_t3_e1-1577983108/submit.txt'
    icdar2013(icdar2013_image_path, icdar2013_output, model, label_dict, device)

    # icdar2015_image_path = '/home/xiekaiyu/ocr/dataset/ICDAR2015WordRecognition/ch4_test_word_images_gt'
    # icdar2015_output = '/home/xiekaiyu/ocr/dataset/ICDAR2015WordRecognition/script_test_ch4_t3_e1-1577983156/submit.txt'
    # icdar2013(icdar2015_image_path, icdar2015_output, model, label_dict, device)


def icdar2013(image_path, output, model, label_dict, device):
    image_names = os.listdir(image_path)
    image_names.sort(key=lambda x: int(x[5:-4]))
    preds = []
    print('Inferencing...')
    for img_name in tqdm(image_names):
        pred = recognize(os.path.join(image_path, img_name), model, label_dict, device)
        preds.append(pred)
    print('Writing to submit.txt ...')
    with open(output, 'w')as f:
        for i in range(len(preds)):
            line = image_names[i] + ', "' + preds[i] + '"\n'
            f.write(line)
    print('Done')


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
