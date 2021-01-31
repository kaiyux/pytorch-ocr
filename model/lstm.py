import torch.nn as nn
from base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, output_channel, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(BiLSTM(output_channel, hidden_size, hidden_size),
                                 BiLSTM(hidden_size, hidden_size, hidden_size))
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, src):
        outs = self.seq(src)
        outs = self.log_softmax(outs)
        return outs


class BiLSTM(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output
