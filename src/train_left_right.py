from seq2seq_model import Sequence
from midi_io_dic_mode import *
from parameters import *
from generate import *
import numpy as np
import torch
from torch import optim
from torch import nn
from keras.layers import Embedding, Dense, Softmax
from keras.models import Sequential
from keras.utils import plot_model
import time
import math
import argparse
import json

device = torch.device("cpu")
from seq2seq_model import DecoderRNN, EncoderRNN
from midi_io_dic_mode import *
from parameters import *
import torch
from torch import optim
from torch import nn
import time
import math
import argparse

device = torch.device("cpu")

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(0)
    encoder_output, encoder_hidden = encoder(input_tensor)  # (time_len, batch_size)
    decoder_input = torch.zeros((1, target_tensor.size(1), target_tensor.size(2)), dtype=torch.float, device=device)
    decoder_hidden = encoder_hidden
    ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
    zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))

    loss = 0

    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) # decoder_output：(1, B, D)
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


def trainIters(train_x, train_y, encoder, decoder, learning_rate=1e-3, batch_size=32):
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_x = torch.tensor(train_x, dtype=torch.float)
    training_y = torch.tensor(train_y, dtype=torch.float)
    criterion = nn.BCELoss(reduction="sum")
    for iter in range(1, training_x.size(0)+1): # iterate each sone
        input_tensor = training_x[iter-1]
        target_tensor = training_y[iter-1]
        loss = 0
        for i in range(0, input_tensor.size(1), batch_size):
            loss += train(input_tensor[:, i: i+batch_size], target_tensor[:, i: i+batch_size],
                          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss

    return print_loss_total




def train_left(x, y, y_token_size=None, x_token_size=None, embed_size=None):
    # one_hot = lambda x, token_size: np.eye(token_size)[x]
    # x = one_hot(x, x_token_size)
    # y = one_hot(y, y_token_size)
    # model = Sequential()
    # # model.add(Embedding(x_token_size, embed_size))
    # model.add(Dense(200, activation='sigmoid'))
    # model.add(Dense(y_token_size, activation='softmax'))
    # # plot_model(model)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
    # model.fit(x, y, epochs=50, batch_size=64)
    mat = np.zeros((x_token_size, y_token_size))
    for i in range(len(x)):
        mat[x[i], y[i]] += 1

    return mat

def get_left(mat, x):
    pred_y = []
    for i in x:
        pred_y.append(np.argmax(mat[i]))
    return pred_y

def combine_left_and_right(left, right, left_corpus, right_corpus):
    assert len(left) == len(right)
    left = dic2pianoroll(left, left_corpus)
    right = dic2pianoroll(right, right_corpus)
    final_chord = (left + right)
    return final_chord

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a MIDI_NET')
    parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    args = parser.parse_args()

    midi_path = "../data/chpn_op7_1.mid"
    model_name = "left_right"
    origin_num_bars = 4
    target_num_bars = 20

    target_length = STAMPS_PER_BAR * target_num_bars
    origin_length = origin_num_bars * STAMPS_PER_BAR

    # step 1 : get the dictionary
    # get_dictionary_of_chord(root_path, two_hand=False)
    right_corpus, right_token_size = load_corpus("../output/chord_dictionary/right-hand.json")

    # step 2 : generated the right-hand music
    pianoroll_data = midiToPianoroll(midi_path, merge=False, velocity=False) # (n_time_stamp, 128, num_track)
    right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
    input_datax, input_datay = createSeqNetInputs([right_track], time_len , 1)

    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = DecoderRNN(input_dim, hidden_dim).to(device)
    if args.load_epoch != 0:
        encoder1.load_state_dict(torch.load('../models/encoder_baseline_' + str(args.load_epoch) + '_Adam1e-3'))
        decoder1.load_state_dict(torch.load('../models/decoder_baseline_' + str(args.load_epoch) + '_Adam1e-3'))

    print("shape of data ", pianoroll_data.shape)
    for i in range(1, args.epoch_number+1):
        loss = trainIters(input_datax, input_datay, encoder1, decoder1)
        print(f'{i} loss {loss}')
        if i % 10 == 0:
            torch.save(encoder1.state_dict(), '../models/encoder_baseline_' + str(i + args.load_epoch) + '_Adam1e-3')
            torch.save(decoder1.state_dict(), '../models/decoder_baseline_' + str(i + args.load_epoch) + '_Adam1e-3')

    # generating
    input_datax = torch.tensor(right_track[:origin_length], dtype=torch.long).unsqueeze(1)

    output, generate_seq = generate(input_datax, encoder1, decoder1, target_length)
    output = [x.item() for x in output]  # real input + generated output
    generate_seq = [x.item() for x in generate_seq]  # generated output

    # step 3 : predict the left-hand music
    left_corpus, left_token_size = load_corpus("../output/chord_dictionary/left-hand.json")
    y = pianoroll2dicMode(left_track, left_corpus)
    x = right_track_input_of_gen
    mat = train_left(x, y, x_token_size=right_token_size, y_token_size=left_token_size, embed_size=100)
    pred_left = get_left(mat, generate_seq)
    chord = combine_left_and_right(pred_left, generate_seq, left_corpus, right_corpus)
    pianorollToMidi(chord, name="right-left-test", velocity=False)







