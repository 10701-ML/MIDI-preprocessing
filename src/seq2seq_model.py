import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, token_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(token_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)

    def forward(self, input):         # [S, N]
        input = self.embedding(input)  # [S, N, E]
        output, hidden = self.gru(input)  # [S, N, H]
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, token_size, embed_size, hidden_size, embedding):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(embed_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden_size, token_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):   # [1, N, H]
        input = self.embedding(input)  # [1, N, E]
        output, hidden = self.gru(input, hidden)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output, hidden


class Sequence(nn.Module):
    def __init__(self, token_size, emb_size, hidden_size):
        super(Sequence, self).__init__()
        self.emb_size = emb_size
        self.Emb = nn.Embedding(token_size, emb_size)
        self.lstm1 = nn.LSTMCell(input_size=emb_size, hidden_size=hidden_size)
        self.lstm2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        # self.drop1 = nn.Dropout(p = 0.2)
        self.softmax = nn.Softmax()


    def forward(self, input):
        return self.out(input)
