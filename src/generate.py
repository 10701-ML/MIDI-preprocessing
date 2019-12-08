from seq2seq_model import DecoderRNN, EncoderRNN
from parameters import *
import torch
from torch import optim
from torch import nn
import time
import math
import argparse
import numpy as np

cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")

def generate(input_tensor, encoder, decoder, target_length):
    encoder.eval()
    decoder.eval()

    encoder_output, encoder_hidden = encoder(input_tensor)
    decoder_input = torch.zeros((1, input_tensor.size(1), input_tensor.size(2)), dtype=torch.float, device=device)
    decoder_hidden = encoder_hidden[1]

    ones = torch.ones(input_tensor.size(1), input_tensor.size(2)).to(device)
    zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2)).to(device)

    generate_seq = []

    for di in range(target_length):
        decoder_output, decoder_hidden,_ = decoder(decoder_input, decoder_hidden, encoder_output)
        decoder_input = torch.where(decoder_output[-1, :, :] > 0.5, ones, zeros)
        decoder_input = decoder_input.unsqueeze(0).detach()  # detach from history as input
        generate_seq.append(decoder_input)

    generate_seq = torch.cat(generate_seq, dim=0)
    output = torch.cat([input_tensor, generate_seq], dim=0)
    print(generate_seq[0])
    return output, generate_seq

