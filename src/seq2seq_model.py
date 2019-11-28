import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output, hidden


class Sequence(nn.Module):
    def __init__(self, token_size, emb_size, hidden_size):
        super(Sequence, self).__init__()
        self.emb_size = emb_size
        self.Emb = nn.Embedding(token_size, emb_size)
        self.lstm1 = nn.LSTM(input_size=emb_size, hidden_size=hidden_size)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, token_size)
        self.drop1 = nn.Dropout(p = 0.2)

    def forward(self, input):
        x = self.Emb(input) 
        x, _ = self.lstm1(x)
        #x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        return x
