from seq2seq_model import Sequence
from midi_io_dic_mode import *
from parameters import *
import torch
from torch import optim
from torch import nn
import time
import math
import argparse
import numpy as np

device = torch.device("cpu")

def generate(input_tensor, model, target_length):
    model.eval()

    print(input_tensor.shape)
    output, hidden = model(input_tensor)
    prediction = torch.argmax(output[-1, :, :], dim=1)
    generate_seq = []
    input = prediction.unsqueeze(0).detach()
    generate_seq.append(prediction)

    for di in range(target_length - 1):
        output, hidden = model(input, hidden)
        prediction = torch.argmax(output[-1, :, :], dim=1)
        input = prediction.unsqueeze(0).detach()  # detach from history as input
        generate_seq.append(prediction)

    generate_seq = torch.cat(generate_seq, dim=0)
    print(generate_seq)
    output = torch.cat([input_tensor, generate_seq.unsqueeze(1)], dim=0).squeeze(1)
    return output, generate_seq

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

    corpus, token_size = load_corpus("../output/chord_dictionary/two-hand.json")  # get the corpus of the chords
    midi_path = next(findall_endswith('.mid', root_path))
    pianoroll_data = midiToPianoroll(midi_path, merge=True, velocity=True)

    piano_data = pianoroll2dicMode(pianoroll_data, corpus)

    input_datax = torch.tensor(piano_data[:args.origin_length], dtype=torch.long).unsqueeze(1)

    model = Sequence(token_size, emb_size, hidden_dim)

    model.load_state_dict(torch.load('../models/dictRNN_' + str(args.load_epoch) + '_Adam1e-3'))

    output, generate_seq = generate(input_datax, model, args.target_length)
    output = [x.item() for x in output]
    generate_seq = [x.item() for x in generate_seq]
    pianorollToMidi(output, name="test_midi", dir="../output/", velocity=False,  # True if the input array contains velocity info (means not binary but continuous)
                    dictionary_dict = corpus) # True if the using the dic mode)
    pianorollToMidi(generate_seq, name="test_midi_gen", dir="../output/", velocity=False,
                    # True if the input array contains velocity info (means not binary but continuous)
                    dictionary_dict=corpus)  # True if the using the dic mode)
