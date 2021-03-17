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
        embeded_tgt = self.embedding(tgt.long()) * math.sqrt(self.d_model)  # (N, T) -> (N, T, E)
        embeded_tgt = embeded_tgt.permute(1, 0, 2)  # (N, T, E) -> (T, N, E)

        # Positional Encoding
        pe = positional_encoding(self.d_model, embeded_tgt.shape[0])  # (T, E)
        pe = pe.repeat(embeded_tgt.shape[1], 1, 1)  # (T, E) -> (N, T, E)
        pe = pe.permute(1, 0, 2)  # (N, T, E) -> (T, N, E)
        pe = pe.to(embeded_tgt.device)
        embeded_tgt += pe

        s = src.shape[0]
        t = embeded_tgt.shape[0]
        n = src.shape[1]

        src_mask = torch.full((s, s), float('-inf')).triu(diagonal=1).to(tgt.device)
        tgt_mask = torch.full((t, t), float('-inf')).triu(diagonal=1).to(tgt.device)

        src_key_padding_mask = torch.full((n, s), False).bool().to(tgt.device)  # we don't mask anything for now
        tgt_key_padding_mask = tgt == 0  # 0 represents UNK in label_dict

        memory_mask = None
        memory_key_padding_mask = None

        out = self.transformer(src, embeded_tgt,
                               src_mask, tgt_mask, memory_mask,
                               src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)  # (T, N, E)
        out = out.permute(1, 0, 2)  # (T, N, E) -> (N, T, E)
        out = self.linear(out)
        out = self.log_softmax(out)
        out = out.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        return out


class TransformerEncoderModel(BaseModel):
    def __init__(self, num_chars, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(in_features=d_model, out_features=num_chars)

    def forward(self, src):
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        out = self.linear(out)  # (S, N, E) -> (S, N, C)
        out = nn.functional.log_softmax(out, dim=2)
        return out


class TransformerDecoderModel(BaseModel):
    def __init__(self, num_chars, d_model=512, nhead=8, num_layers=6, norm=None):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=num_chars, embedding_dim=d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm=norm)
        self.linear = nn.Linear(in_features=d_model, out_features=num_chars)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, src, tgt):
        # Embedding
        embeded_tgt = self.embedding(tgt.long()) * math.sqrt(self.d_model)  # (N, T) -> (N, T, E)
        embeded_tgt = embeded_tgt.permute(1, 0, 2)  # (N, T, E) -> (T, N, E)

        # Positional Encoding
        pe = positional_encoding(self.d_model, embeded_tgt.shape[0])  # (T, E)
        pe = pe.repeat(embeded_tgt.shape[1], 1, 1)  # (T, E) -> (N, T, E)
        pe = pe.permute(1, 0, 2)  # (N, T, E) -> (T, N, E)
        pe = pe.to(embeded_tgt.device)
        embeded_tgt += pe

        t = embeded_tgt.shape[0]
        tgt_mask = torch.full((t, t), float('-inf')).triu(diagonal=1).to(tgt.device)
        tgt_key_padding_mask = tgt == 0  # 0 represents UNK in label_dict

        if self.training:
            out = self.transformer_decoder(embeded_tgt, src,
                                           tgt_mask, None,
                                           tgt_key_padding_mask, None)  # (T, N, E)
        else:
            out = self.transformer_decoder(embeded_tgt, src,
                                           None, None,
                                           tgt_key_padding_mask, None)  # (T, N, E)

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


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
