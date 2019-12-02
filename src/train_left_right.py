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

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(train_x, train_y, model, token_size, learning_rate=1e-3, batch_size=32):
    model.train()
    print_loss_total = 0  # Reset every print_every
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    training_x = torch.tensor(train_x, dtype=torch.long) # (num_of_songs, num_of_samples, time_len)
    training_y = torch.tensor(train_y, dtype=torch.long)
    for iter in range(1, training_x.size(0)+1): # iterate each sone
        optimizer.zero_grad()
        input_tensor = training_x[iter-1]
        target_tensor = training_y[iter-1]
        loss = 0
        for i in range(0, input_tensor.size(1), batch_size):
            input_tensor_batch = input_tensor[:, i: i+batch_size]   #(time_len, batch_size)
            target_tensor_batch = target_tensor[:, i: i+batch_size]  #(time_len, batch_size)
            target_size = input_tensor_batch.size(0)
            input = torch.zeros_like(input_tensor_batch[0:1, :])
            hidden = None
            for di in range(target_size):
                output, hidden = model(input, hidden)
                target = target_tensor_batch[di, :]
                output = output.reshape(-1, token_size)
                loss += criterion(output, target)
                if torch.rand(1)[0] > threshold:
                    input = target_tensor_batch[di].unsqueeze(0)

                else:
                    input = torch.argmax(output, dim=1)
                    input = input.unsqueeze(0).detach()

        loss.backward()
        optimizer.step()
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
    midi_path = "../data/chpn_op7_1.mid"
    model_name = "left_right"
    load_epoch = 1000
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
    input_datax, input_datay = createSeqNetInputs([right_track], time_len , 1, right_corpus)
    model = Sequence(right_token_size, emb_size, hidden_dim)
    print("shape of data ", pianoroll_data.shape)
    epoch = 500
    for i in range(1, epoch+1):
        loss = trainIters(input_datax, input_datay, model, token_size=right_token_size)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(model.state_dict(), f'../models/{model_name}_' + str(i + load_epoch) + '_Adam1e-3')

    right_track_input_of_gen = pianoroll2dicMode(right_track, right_corpus)
    input_datax = torch.tensor(right_track_input_of_gen[:origin_length], dtype=torch.long).unsqueeze(1)

    output, generate_seq = generate(input_datax, model, target_length)
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







