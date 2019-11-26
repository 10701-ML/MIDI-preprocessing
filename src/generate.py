from seq2seq_model import DecoderRNN, EncoderRNN
from midi_io import midiToPianoroll, seqNetOutToPianoroll, pianorollToMidi
from parameters import *
import torch
from torch import optim
from torch import nn
import time
import math
import argparse
import numpy as np

device = torch.device("cpu")

def generate(input_tensor, encoder, decoder, target_length):
    encoder.eval()
    decoder.eval()

    encoder_output, encoder_hidden = encoder(input_tensor)
    decoder_input = torch.zeros((1, input_tensor.size(1), input_tensor.size(2)), dtype=torch.float, device=device)
    decoder_hidden = encoder_hidden

    ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
    zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))

    generate_seq = []

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        print(decoder_output.shape)
        decoder_input = torch.where(decoder_output[-1, :, :] > 0.5, ones, zeros)
        decoder_input = decoder_input.unsqueeze(0).detach()  # detach from history as input
        generate_seq.append(decoder_input)

    generate_seq = torch.cat(generate_seq, dim=0)
    output = torch.cat([input_tensor, generate_seq], dim=0)
    return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate a MIDI')
    parser.add_argument('-t', '--target_length', type=int, help='the target length')
    parser.add_argument('-o', '--origin_length', type=int, help='the origin length')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    args = parser.parse_args()

    if args.origin_length <= 0:
        print("invalid origin length")
        exit()

    if args.target_length <= 0:
        print("invalid target length")
        exit()

    if args.load_epoch <= 0:
        print("invalid load epoch")
        exit()

    piano_data = midiToPianoroll(path, debug=True)
    print("shape of data ", piano_data.shape)

    input_datax = torch.from_numpy(piano_data[0:args.origin_length, :]).unsqueeze(1).float()

    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = DecoderRNN(input_dim, hidden_dim).to(device)

    encoder1.load_state_dict(torch.load('../models/encoder_baseline_' + str(args.load_epoch) + '_Adam1e-3'))
    decoder1.load_state_dict(torch.load('../models/decoder_baseline_' + str(args.load_epoch) + '_Adam1e-3'))

    output = generate(input_datax, encoder1, decoder1, args.target_length)
    piano_roll = seqNetOutToPianoroll(output)
    pianorollToMidi(piano_roll, "../output/test.mid")
