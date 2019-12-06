from seq2seq_model import DecoderRNN, EncoderRNN
from parameters import *
import torch
from torch import optim
from torch import nn
import time
import math
import argparse
import numpy as np

device = torch.device("cpu")
#
# def generate(input_tensor, encoder, decoder, target_length, random=False, random_interval=0):
#     encoder.eval()
#     decoder.eval()
#     encoder_output, encoder_hidden = encoder(input_tensor)
#     decoder_input = torch.zeros((1, input_tensor.size(1), input_tensor.size(2)), dtype=torch.float, device=device)
#     decoder_hidden = encoder_hidden
#
#     ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
#     zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))
#
#     generate_seq = []
#
#     for di in range(target_length):
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#         decoder_input = torch.where(decoder_output[-1, :, :] > 0.5, ones, zeros)
#         decoder_input = decoder_input.unsqueeze(0).detach()  # detach from history as input
#         if random and di:
#             pass
#         generate_seq.append(decoder_input)
#
#     generate_seq = torch.cat(generate_seq, dim=0)
#     output = torch.cat([input_tensor, generate_seq], dim=0)
#     return output, generate_seq


# def generate(input_tensor, encoder, decoder, target_length, random=False, random_interval=0):
#     encoder.eval()
#     decoder.eval()
#     encoder_output, encoder_hidden = encoder(input_tensor)
#     encoder_hidden = (torch.zeros_like(encoder_hidden[0]), torch.zeros_like(encoder_hidden[1]))
#     decoder_output, decoder_hidden = decoder(input_tensor, encoder_hidden)
#     ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
#     zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))
#     decoder_input = torch.where(decoder_output[-1, :, :] > 0.5, ones, zeros).unsqueeze(0).detach()
#     generate_seq = []
#
#     for di in range(target_length):
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#         decoder_input = torch.where(decoder_output[-1, :, :] > 0.5, ones, zeros)
#         decoder_input = decoder_input.unsqueeze(0).detach()  # detach from history as input
#         if random and di:
#             pass
#         generate_seq.append(decoder_input)
#
#     generate_seq = torch.cat(generate_seq, dim=0)
#     output = torch.cat([input_tensor, generate_seq], dim=0)
#     return output, generate_seq

def generate(input_tensor, encoder, decoder, target_length, random=False, random_interval=0):
    encoder.eval()
    decoder.eval()
    encoder_output, encoder_hidden = encoder(input_tensor)
    decoder_input = input_tensor[-1, :, :].unsqueeze(0).detach()
    decoder_hidden = encoder_hidden

    ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
    zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))

    generate_seq = []

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        if random and ((di + 1) % random_interval) == 0:
            noise = torch.randn_like(decoder_output) / 5
            decoder_output +=noise
        decoder_input = torch.where(decoder_output[-1, :, :] > 0.5, ones, zeros)
        decoder_input = decoder_input.unsqueeze(0).detach()  # detach from history as input


        generate_seq.append(decoder_input)

    generate_seq = torch.cat(generate_seq, dim=0)
    output = torch.cat([input_tensor, generate_seq], dim=0)
    return output, generate_seq

# def generate(input_tensor, encoder, decoder, target_length, random=False, random_interval=0):
#     encoder.eval()
#     decoder.eval()
#     encoder_output, encoder_hidden = encoder(input_tensor)
#     decoder_hidden = encoder_hidden
#
#     ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
#     zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))
#
#     generate_seq = []
#     for di in range(len(input_tensor)):
#         decoder_input = input_tensor[-1, :, :].unsqueeze(0)
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#         generate_seq.append(torch.where(decoder_output[-1, :, :] > 0.5, ones, zeros))
#
#     generate_seq = torch.cat(generate_seq, dim=0)
#     output = torch.cat([input_tensor[:, 0, :], generate_seq], dim=0)
#     return output, generate_seq
