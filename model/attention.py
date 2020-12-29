import math
import torch
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
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=num_chars, embedding_dim=d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation=activation, custom_encoder=custom_encoder,
                                          custom_decoder=custom_decoder)
        self.linear = nn.Linear(in_features=d_model, out_features=num_chars)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, src, tgt):
        # Embedding
        tgt = self.embedding(tgt.long())  # (N, T) -> (N, T, E)
        tgt = tgt.permute(1, 0, 2)  # (N, T, E) -> (T, N, E)

        # Positional Encoding
        pe = positional_encoding(self.d_model, tgt.shape[0])  # (T, E)
        pe = pe.repeat(tgt.shape[1], 1, 1)  # (T, E) -> (N, T, E)
        pe = pe.permute(1, 0, 2)  # (N, T, E) -> (T, N, E)
        pe = pe.to(tgt.device)
        tgt += pe

        # TODO:mask
        out = self.transformer(src, tgt)  # (T, N, E)
        out = out.permute(1, 0, 2)  # (T, N, E) -> (N, T, E)
        out = self.linear(out)
        out = self.log_softmax(out)
        out = out.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        return out


def positional_encoding(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
