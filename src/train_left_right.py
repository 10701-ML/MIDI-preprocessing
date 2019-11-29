from seq2seq_model import Sequence
from midi_io_dic_mode import *
from parameters import *
from generate import *
import torch
from torch import optim
from torch import nn
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
        # torch.random.
        for i in range(0, input_tensor.size(1), batch_size):
            input_tensor_batch = input_tensor[:, i : i+batch_size] #(time_len, batch_size)
            target_tensor_batch = target_tensor[:, i: i+batch_size]
            output, _ = model(input_tensor_batch)
            output = output.reshape(-1, token_size)
            target_tensor_batch = target_tensor_batch.reshape(-1)
            loss += criterion(output, target_tensor_batch)

        loss.backward()
        optimizer.step()
        print_loss_total += loss

    return print_loss_total

def train_right():


    return model

def generate_right(model):
    pass

if __name__ == "__main__":
    midi_path = "../data/chpn_op7_1.mid"
    model_name = "left_right"
    target_length = 100

    # step 1 : get the dictionary
    # get_dictionary_of_chord(root_path, two_hand=False)
    right_corpus, right_token_size = load_corpus("../output/chord_dictionary/right-hand.json")
    pianoroll_data = midiToPianoroll(midi_path, merge=False, velocity=False) # (n_time_stamp, 128, num_track)
    right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
    input_datax, input_datay = createSeqNetInputs([right_track], time_len , output_len, right_corpus)
    model = Sequence(right_token_size, emb_size, hidden_dim)
    print("shape of data ", pianoroll_data.shape)
    epoch = 1
    for i in range(1, epoch+1):
        loss = trainIters(input_datax, input_datay, model, token_size=right_token_size)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(model.state_dict(), f'../models/{model_name}_' + str(i + args.load_epoch) + '_Adam1e-3')

    output, generate_seq = generate(input_datax, model, target_length)
    output = [x.item() for x in output]
    generate_seq = [x.item() for x in generate_seq]



