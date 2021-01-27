import argparse
import torch
from torchvision import transforms
from PIL import Image
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.datasets import get_label_dict


def main():
    image_path = '/home/stu7/workspace/ocr/dataset/all/ICDAR2013WordRecognition/fake/word_1.png'
    resume = '/home/stu7/workspace/ocr/pytorch-ocr/recog_model/models/OCR/0127_161757/checkpoint-epoch100.pth'
    label_dict = '/home/stu7/workspace/ocr/pytorch-ocr/label_dicts/label_dict_en.txt'
    logger = config.get_logger('inference')

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

    img = Image.open(image_path).convert("RGB")
    tgt_width = 224
    tgt_height = 64

    # width, height = img.size
    # if width < tgt_width:
    #     reshape_width = tgt_height * (width / height)
    #     img = img.resize([int(reshape_width), int(tgt_height)])
    #     # padding
    #     pad_width = tgt_width - img.size[0]
    #     if pad_width < 0:
    #         img = img.resize([int(tgt_width), int(tgt_height)])
    #     else:
    #         pad = transforms.Compose([transforms.Pad(padding=(0, 0, pad_width, 0))])
    #         img = pad(img)
    # else:
    #     img = img.resize([int(tgt_width), int(tgt_height)])
    img = img.resize([int(tgt_width), int(tgt_height)])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = transform(img).unsqueeze(0).to(device)
    output = model(img)

    _, ind2ch = get_label_dict(label_dict)
    outputs = output.argmax(dim=2).permute(1, 0).cpu().numpy().tolist()
    for i in range(len(outputs)):
        pred = []
        for ch in outputs[i]:
            pred.append(ind2ch[ch])
        print('====================')
        print(pred)
        print('====================')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main()
