import numpy as np
import torch
from generate import *
from midi_io_dic_mode import *
from midi_io_musegan import findall_endswith, make_sure_path_exists
from parameters import *
import os
device = torch.device("cpu")
from seq2seq_model import DecoderRNN, EncoderRNN, AttnDecoderRNN
from train_left_right import trainIters, predict
from midi_io_dic_mode import *
from parameters import *
import torch
from torch import optim
from torch import nn
import argparse
cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(0)
    encoder_output, encoder_hidden = encoder(input_tensor)  # (time_len, batch_size, D)
    decoder_input = torch.zeros((1, target_tensor.size(1), target_tensor.size(2)), dtype=torch.float, device=device)
    decoder_hidden = encoder_hidden[1]
    ones = torch.ones(input_tensor.size(1), input_tensor.size(2)).to(device)
    zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2)).to(device)
    loss = 0
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output) # decoder_output：(1, B, D)
        loss += criterion(decoder_output[0], target_tensor[di])
        if torch.rand(1)[0] > threshold:
            decoder_input = target_tensor[di].unsqueeze(0)

        else:
            decoder_input = torch.where(decoder_output[0, :, :] > 0.5, ones, zeros)
            decoder_input = decoder_input.unsqueeze(0).detach()  # detach from history as input

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def trainIters(train_x, train_y, encoder, decoder, learning_rate=1e-4, batch_size=32):
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.BCELoss(reduction="sum")
    for iter in range(1, len(train_x)+1): # iterate each sone
        input_tensor = train_x[iter-1]
        target_tensor = train_y[iter-1]
        input_tensor = torch.tensor(input_tensor, dtype=torch.float).to(device)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float).to(device)
        loss = 0
        for i in range(0, input_tensor.size(1), batch_size):
            loss += train(input_tensor[:, i: i+batch_size], target_tensor[:, i: i+batch_size],
                          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss

    return print_loss_total


def train_mul(args):
    model_name = "left_right_mul-beat8"
    origin_num_bars = 10
    target_num_bars = 20
    target_length = STAMPS_PER_BAR * target_num_bars
    origin_length = origin_num_bars * STAMPS_PER_BAR
    right_tracks = []
    left_tracks = []
    for midi_path in findall_endswith(".mid", "../data/"):
        pianoroll_data = midiToPianoroll(midi_path, merge=False, velocity=False)  # (n_time_stamp, 128, num_track)
        if pianoroll_data.shape[2] != 1:
            right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
            right_tracks.append(right_track)
            left_tracks.append(left_track)

    input_datax, input_datay = createSeqNetInputs(right_tracks, time_len, output_len)
    #print(input_datax.shape)
    encoder1 = EncoderRNN(time_len, input_dim, hidden_dim).to(device)
    decoder1 = AttnDecoderRNN(input_dim, hidden_dim, dropout_p=0.1, max_length=time_len).to(device)
    if args.load_epoch != 0:
        encoder1.load_state_dict(torch.load(f'../models/mul_encoder_{model_name}_' + str(args.load_epoch)))
        decoder1.load_state_dict(torch.load(f'../models/mul_decoder_{model_name}_' + str(args.load_epoch)))

    for i in range(1, args.epoch_number + 1):
        loss = trainIters(input_datax, input_datay, encoder1, decoder1)
        print(f'{i} loss {loss}')
        if i % 10 == 0:
            torch.save(encoder1.state_dict(), f'../models/mul_encoder_{model_name}_' + str(i + args.load_epoch))
            torch.save(decoder1.state_dict(), f'../models/mul_decoder_{model_name}_' + str(i + args.load_epoch))
    midi_path = "../test/chpn_op66.mid"
    predict(midi_path, origin_length, encoder1, decoder1, target_length, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a MIDI_NET')
    parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate', default=0.001)
    args = parser.parse_args()
    train_mul(args)







