from model.model import RecognizeModel
from benchmarks.models import NRTR
import torch
from time import time


def forward_time_benchmark(models, device='gpu'):
    trials = 100
    batch_size = 1

    for name in models.keys():
        time_cost = 0
        for i in range(trials):
            img = torch.rand(size=(batch_size, 3, 64, 320), dtype=torch.float)
            model = models[name]
            if device == 'gpu':
                model = model.to('cuda')
                img = img.to('cuda')
            model.eval()

            tik = time()
            model(img)
            tok = time()

            time_cost += (tok - tik)
            del model

        avg_time = time_cost / trials
        print(f'{name} forward time: {avg_time * 1000:.2f}ms per batch. (batch_size={batch_size})')


def main():
    model = RecognizeModel(num_chars=99, d_model=512, nhead=8, num_layers=6,
                           backbone='ResNet34', head='TransformerEncoder')
    nrtr = NRTR()

    models = {
        'ResNet34-TransformerEncoder': model,
        'NRTR': nrtr
    }

    forward_time_benchmark(models)


if __name__ == '__main__':
    main()
