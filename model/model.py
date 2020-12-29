import torch
from base import BaseModel
from .attention import TransformerModel
from .backbone import ShuffleNetV2


class RecognizeModel(BaseModel):
    def __init__(self, num_chars, d_model):
        super().__init__()
        self.backbone = ShuffleNetV2()
        self.transformer = TransformerModel(num_chars, d_model=d_model, nhead=16, num_encoder_layers=12)

    def forward(self, img, transcription):
        features = self.backbone(img)
        N, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(torch.Size([N, C, H * W]))  # (N, E, S)
        features = features.permute(2, 0, 1)  # (N, E, S) -> (S, N, E)

        outs = self.transformer(features, transcription)
        return outs
