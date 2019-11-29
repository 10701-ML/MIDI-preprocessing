from seq2seq_model import Sequence
from midi_io_dic_mode import *
from parameters import *
import torch
from torch import optim
from torch import nn
import time
import math
import argparse
import json

from midi_io_dic_mode import *

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


def trainIters(train_x, train_y, model, max_length, learning_rate=1e-3, batch_size=32):
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
            input_tensor_batch = input_tensor[:, i : i+batch_size] #(batch_size, time_len)
            target_tensor_batch = target_tensor[:, i: i+batch_size]
            output, _ = model(input_tensor_batch)
            output = output.reshape(-1, token_size)
            target_tensor_batch = target_tensor_batch.reshape(-1)
            loss += criterion(output, target_tensor_batch)
        
        loss.backward()
        optimizer.step()
        
        print_loss_total += loss
    return print_loss_total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a MIDI_NET')
    parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    args = parser.parse_args()

    if args.epoch_number <= 0:
        print("invalid epoch number")
        exit()
    epoch = args.epoch_number

    get_dictionary_of_chord(root_path, two_hand=True)
    corpus, token_size = load_corpus("../output/chord_dictionary/two-hand.json") # get the corpus of the chords
    model = Sequence(token_size, emb_size, hidden_dim)
    if args.load_epoch != 0:
        model.load_state_dict(torch.load('../models/dictRNN_' + str(args.load_epoch) + '_Adam1e-3'))

    root_path = "../data/"
    midi_path = next(findall_endswith('.mid', root_path))
    pianoroll_data = midiToPianoroll(midi_path, merge=True, velocity=True)
    input_datax, input_datay = createSeqNetInputs([pianoroll_data], 200, 200, corpus)

    print("shape of data ", pianoroll_data.shape)

    for i in range(1, epoch+1):
        loss = trainIters(input_datax, input_datay, model, max_length=4000)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(model.state_dict(), '../models/dictRNN_' + str(i + args.load_epoch) + '_Adam1e-3')

