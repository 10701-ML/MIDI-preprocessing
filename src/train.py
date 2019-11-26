from seq2seq_model import DecoderRNN, EncoderRNN
from midi_io import createSeqNetInputs, midiToPianoroll
from parameters import *
import torch
from torch import optim
from torch import nn
import time
import math
import argparse

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


def trainIters(train_x, train_y, encoder, decoder, max_length, print_every=1, learning_rate=1e-3, batch_size=32):
    start = time.time()
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
        # torch.random.
        for i in range(0, input_tensor.size(1), batch_size):
            loss += train(input_tensor[:, i: i+batch_size, :], target_tensor[:, i: i+batch_size, :],
                          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)

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

    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = DecoderRNN(input_dim, hidden_dim).to(device)

    if args.load_epoch != 0:
        encoder1.load_state_dict(torch.load('../models/encoder_baseline_' + str(args.load_epoch) + '_Adam1e-3'))
        decoder1.load_state_dict(torch.load('../models/decoder_baseline_' + str(args.load_epoch) + '_Adam1e-3'))


    piano_data = midiToPianoroll(path, debug=True)
    print("shape of data ", piano_data.shape)

    input_datax, input_datay = createSeqNetInputs([piano_data], time_len, output_len)

    for i in range(1, epoch+1):
        loss = trainIters(input_datax, input_datay, encoder1, decoder1, max_length=4000)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(encoder1.state_dict(), '../models/encoder_baseline_' + str(i + args.load_epoch) + '_Adam1e-3')
            torch.save(decoder1.state_dict(), '../models/decoder_baseline_' + str(i + args.load_epoch) + '_Adam1e-3')

