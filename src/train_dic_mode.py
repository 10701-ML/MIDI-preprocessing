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

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(0)
    encoder_output, encoder_hidden = encoder(input_tensor)
    decoder_input = torch.zeros((1, target_tensor.size(1), target_tensor.size(2)), dtype=torch.float, device=device)
    decoder_hidden = encoder_hidden
    ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
    zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))

    loss = 0

    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
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


def trainIters(train_x, train_y, model, max_length, learning_rate=1e-3, batch_size=32):
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_x = torch.tensor(train_x, dtype=torch.float) # (num_of_songs, num_of_samples, time_len)
    training_y = torch.tensor(train_y, dtype=torch.float)
    criterion = nn.CrossEntropyLoss()
    for iter in range(1, training_x.size(0)+1): # iterate each sone
        input_tensor = training_x[iter-1]
        target_tensor = training_y[iter-1]
        loss = 0
        # torch.random.
        for i in range(0, input_tensor.size(0), batch_size):
            input_tensor = input_tensor[i: i+batch_size, :] #(batch_size, time_len)
            target_tensor = target_tensor[i: i+batch_size, :]
            optimizer.zero_grad()
            output = model(input_tensor)
            loss += criterion(output, target_tensor)

        print_loss_total += loss
    return print_loss_total

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='train a MIDI_NET')
    # parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    # parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    # args = parser.parse_args()

    # if args.epoch_number <= 0:
    #     print("invalid epoch number")
    #     exit()
    # epoch = args.epoch_number
    epoch = 100

    corpus, token_size = load_corpus("../output/chord_dictionary/two-hand.json") # get the corpus of the chords
    model = Sequence(token_size, emb_size, hidden_dim)
    # if args.load_epoch != 0:
    #     model.load_state_dict(torch.load('../models/deepjazz_baseline_' + str(args.load_epoch) + '_Adam1e-3'))

    root_path = "../data/"
    midi_path = next(findall_endswith('.mid', root_path))
    pianoroll_data = midiToPianoroll(midi_path, corpus)
    input_datax, input_datay = createSeqNetInputs([pianoroll_data], 5, 5, corpus)

    print("shape of data ", pianoroll_data.shape)

    for i in range(1, epoch+1):
        loss = trainIters(input_datax, input_datay, model, max_length=4000)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(model.state_dict(), '../models/deepjazz_baseline_' + str(i + args.load_epoch) + '_Adam1e-3')

