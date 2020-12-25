import torch.nn as nn
from base import BaseModel


class TransformerModel(BaseModel):
    def __init__(self,
                 num_chars,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 custom_encoder=None,
                 custom_decoder=None):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation=activation, custom_encoder=custom_encoder,
                                          custom_decoder=custom_decoder)
        self.linear = nn.Linear(in_features=d_model, out_features=num_chars)
        self.softmax = nn.Softmax()

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)  # (T, N, E)
        out = out.permute(1, 0, 2)  # (T, N, E) -> (N, T, E)
        out = self.linear(out)
        out = self.softmax(out)
        return out
