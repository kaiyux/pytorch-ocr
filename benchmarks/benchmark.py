from model.model import RecognizeModel
from benchmarks.models import NRTR
import torch
import numpy as np
from time import time


def forward_time_benchmark(models, device='gpu'):
    trials = 100
    batch_size = 4

    for name in models.keys():
        print(f'{name}')
        time_cost = 0
        model = models[name]
        for i in range(trials):
            img = torch.rand(size=(batch_size, 3, 64, 320), dtype=torch.float)

            if device == 'gpu':
                model = model.to('cuda')
                img = img.to('cuda')
            model.eval()

            tik = time()
            model(img)
            tok = time()

            time_cost += (tok - tik)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        avg_time = time_cost / trials
        print(f'Forward time: {avg_time * 1000:.2f}ms per batch. (batch_size={batch_size})')
        print(f'Trainable parameters: {params}\n')
        del model


def main():
    model = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                           backbone='ResNet34', head='TransformerEncoder')
    resnet18_transformer_encoder = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                                  backbone='ResNet18', head='TransformerEncoder')
    shufflenet_transformer_encoder = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                                    backbone='ShuffleNet', head='TransformerEncoder')
    resnet34_lstm = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                   backbone='ResNet34', head='LSTM')
    nrtr = NRTR()

    models = {
        'ResNet34-TransformerEncoder': model,
        'ResNet18-TransformerEncoder': resnet18_transformer_encoder,
        'ShuffleNet-TransformerEncoder': shufflenet_transformer_encoder,
        'ResNet34-LSTM': resnet34_lstm,
        'NRTR': nrtr
    }

    forward_time_benchmark(models)


if __name__ == '__main__':
    main()
