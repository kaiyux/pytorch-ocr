import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .attention import TransformerModel
from .backbone import ShuffleNetV2


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class RecognizeModel(BaseModel):
    def __init__(self, num_chars, d_model):
        super().__init__()
        self.backbone = ShuffleNetV2()
        self.embedding = nn.Embedding(num_embeddings=num_chars, embedding_dim=d_model)
        self.transformer = TransformerModel(num_chars, d_model=d_model, nhead=16, num_encoder_layers=12)

    def forward(self, img, transcription):
        features = self.backbone(img)
        N, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(torch.Size([N, C, H * W]))  # (N, E, S)
        features = features.permute(2, 0, 1)  # (N, E, S) -> (S, N, E)

        transcription = self.embedding(transcription.long())  # (N, T) -> (N, T, E)
        transcription = transcription.permute(1, 0, 2)  # (N, T, E) -> (T, N, E)

        outs = self.transformer(features, transcription)
        return outs
