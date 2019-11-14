from seq2seq_model import DecoderRNN, EncoderRNN
from midi_io import createSeqNetInputs, midiToPianoroll
import torch
from torch import optim
from torch import nn
import time
import math

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

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor(target_length, device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(train_x, train_y, encoder, decoder, max_length, print_every=5, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_x = torch.tensor(train_x)
    training_y = torch.tensor(train_y)
    criterion = nn.NLLLoss()
    batch_size = 32
    for iter in range(1, training_x.size(1), batch_size):
        input_tensor = training_x[:, iter-1: iter+batch_size-1, :]
        target_tensor = training_y[:, iter-1: iter+batch_size-1, :]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / training_x.size(1)),
                                         iter, iter / training_x.size(1) * 100, print_loss_avg))


if __name__ == "__main__":
    path = "./data/chp_op18.mid"
    piano_data = midiToPianoroll(path, debug=True)
    print("shape of data ", piano_data.shape)
    time_len = 5
    output_len = 5
    hidden_dim = 256
    input_dim = 88
    output_dim = 88
    epoch = 10
    input_datax, input_datay = createSeqNetInputs([piano_data], time_len, output_len)
    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = DecoderRNN(hidden_dim, output_dim)
    for i in range(1, epoch+1):
        trainIters(input_datax, input_datay, encoder1, decoder1, time_len)
