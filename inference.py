import argparse
import torch
from torchvision import transforms
from PIL import Image
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.datasets import get_label_dict


def main():
    image_path = '/home/stu7/workspace/ocr/dataset/all/ICDAR2015WordRecognition/word_1.png'
    resume = '/home/stu7/workspace/ocr/pytorch-ocr/recog_model/models/OCR/0126_124319/checkpoint-epoch6.pth'
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

    img = Image.open(image_path)
    tgt_width = 224
    tgt_height = 64
    img = img.resize([int(tgt_width), int(tgt_height)])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = transform(img).unsqueeze(0).to(device)
    transcription = torch.LongTensor([1.]).unsqueeze(0).to(device)  # SOS
    output = model(img, transcription)

    output = output.argmax(dim=1).cpu().numpy().tolist()
    _, ind2ch = get_label_dict(label_dict)

    for sentence in output:
        line = []
        for ch in sentence:
            line.append(ind2ch[ch])
        print(line)


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
