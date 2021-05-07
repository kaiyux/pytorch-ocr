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

    # ablation study
    # 1. nhead & nlayers
    nhead1_nlayer1_model = RecognizeModel(num_chars=99, d_model=512, nhead=1, num_layers=1,
                                          backbone='ResNet34', head='TransformerEncoder')
    nhead4_nlayer1_model = RecognizeModel(num_chars=99, d_model=512, nhead=4, num_layers=1,
                                          backbone='ResNet34', head='TransformerEncoder')
    nhead8_nlayer1_model = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=1,
                                          backbone='ResNet34', head='TransformerEncoder')
    nhead16_nlayer1_model = RecognizeModel(num_chars=99, d_model=512, nhead=16, num_layers=1,
                                           backbone='ResNet34', head='TransformerEncoder')
    nhead8_nlayer2_model = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=2,
                                          backbone='ResNet34', head='TransformerEncoder')

    # 2. backbone
    shufflenet_transformer_encoder = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                                    backbone='ShuffleNet', head='TransformerEncoder')
    resnet18_transformer_encoder = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                                  backbone='ResNet18', head='TransformerEncoder')
    resnet50_transformer_encoder = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                                  backbone='ResNet50', head='TransformerEncoder')

    # 3. head
    resnet34_transformer_decoder = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                                  backbone='ResNet34', head='TransformerDecoder')
    resnet34_lstm = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                                   backbone='ResNet34', head='LSTM')

    # 4. SOTA
    nrtr = NRTR()

    models = {
        'ResNet34-TransformerEncoder': model,

        'nhead1_nlayer1_model': nhead1_nlayer1_model,
        'nhead1_nlayer4_model': nhead4_nlayer1_model,
        'nhead1_nlayer8_model': nhead8_nlayer1_model,
        'nhead1_nlayer16_model': nhead16_nlayer1_model,
        'nhead8_nlayer2_model': nhead8_nlayer2_model,

        'ResNet18-TransformerEncoder': resnet18_transformer_encoder,
        'ResNet50-TransformerEncoder': resnet50_transformer_encoder,
        'ShuffleNet-TransformerEncoder': shufflenet_transformer_encoder,

        'ResNet34-TransformerDecoder': resnet34_transformer_decoder,
        'ResNet34-LSTM': resnet34_lstm,

        'NRTR': nrtr
    }

    forward_time_benchmark(models)


if __name__ == '__main__':
    main()
