import torch
from base import BaseModel
from .attention import TransformerModel, TransformerEncoderModel, TransformerDecoderModel
from .lstm import LSTMModel
from .backbone import ShuffleNetV2, TinyNet, resnet18, resnet34, resnet50

arch_backbones = {
    'ShuffleNet': ShuffleNetV2,
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50
}

arch_heads = {
    'LSTM': LSTMModel,
    'Transformer': TransformerModel,
    'TransformerEncoder': TransformerEncoderModel,
    'TransformerDecoder': TransformerDecoderModel
}


class RecognizeModel(BaseModel):
    def __init__(self, num_chars, d_model, nhead, num_layers,
                 backbone='ResNet18', head='TransformerEncoder'):
        super().__init__()
        assert backbone in arch_backbones.keys(), 'Invalid backbone.'
        assert head in arch_heads.keys(), 'Invalid head.'
        if head in ['Transformer', 'TransformerDecoder']:
            self.has_decoder = True
        else:
            self.has_decoder = False

        self.backbone = arch_backbones[backbone]()

        if head == 'LSTM':
            self.head = LSTMModel(d_model, num_chars)
        else:
            self.head = arch_heads[head](num_chars, d_model, nhead, num_layers)

    def forward(self, img, transcription=None):
        features = self.backbone(img)

        N, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(torch.Size([N, C, H * W]))  # (N, E, S)
        features = features.permute(2, 0, 1)  # (N, E, S) -> (S, N, E)

        if self.has_decoder:
            outs = self.head(features, transcription)
        else:
            outs = self.head(features)
        return outs
