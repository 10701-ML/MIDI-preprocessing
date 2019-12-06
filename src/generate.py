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
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_input = torch.zeros((1, 1), dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden
    generate_seq = []
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        prediction = torch.argmax(decoder_output[0], dim=1)
        decoder_input = prediction.unsqueeze(0).detach()  # detach from history as input
        generate_seq.append(decoder_input)

    generate_seq = torch.cat(generate_seq, dim=0)
    output = torch.cat([input_tensor, generate_seq], dim=0)
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

    corpus, token_size = load_corpus("../output/chord_dictionary/two-hand.json")
    midi_path = next(findall_endswith('.mid', root_path))
    piano_data = midiToPianoroll(path, merge=True, velocity=True)
    dic_data = pianoroll2dicMode(piano_data, corpus)
    #pianorollToMidi(dic_data, "../output/test_attention_origin.mid", velocity=False, dictionary_dict=corpus)
    # print("shape of data ", dic_data.shape)

    input_datax = torch.tensor(dic_data[500:500 + args.origin_length]).unsqueeze(1).long()
    # print(input_datax.shape)

    #encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    #decoder1 = DecoderRNN(input_dim, hidden_dim).to(device)
    encoder1 = EncoderRNN(token_size, emb_size, hidden_dim).to(device)
    attn_decoder1 = AttnDecoderRNN(token_size, emb_size, hidden_dim, encoder1.embedding, dropout_p=0.1, max_length=args.origin_length).to(device)

    encoder1.load_state_dict(torch.load('../models/encoder_dict_' + str(args.load_epoch) + '_Adam1e-3'))
    attn_decoder1.load_state_dict(torch.load('../models/decoder_dict_' + str(args.load_epoch) + '_Adam1e-3'))

    output, output_gen = generate(input_datax, encoder1, attn_decoder1, args.target_length)
    output = [i.item() for i in output]
    output_gen = [i.item() for i in output_gen]

    pianorollToMidi(output, "../output/test_attention_" + str(args.load_epoch) + ".mid", velocity=False, dictionary_dict=corpus)
    pianorollToMidi(output_gen, "../output/test_attention_gen_.mid" + str(args.load_epoch) + ".mid", velocity=False, dictionary_dict=corpus)

