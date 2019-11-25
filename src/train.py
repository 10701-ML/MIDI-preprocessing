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

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(0)
    encoder_output, encoder_hidden = encoder(input_tensor)
    decoder_input = torch.zeros((1, target_tensor.size(1), target_tensor.size(2)), dtype=torch.float, device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True
    loss = 0
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output[0], target_tensor[di])
            decoder_input = target_tensor[di].unsqueeze(0)  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output[0], target_tensor[di])
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

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
        for i in range(0, input_tensor.size(0), batch_size):
            loss += train(input_tensor[:, i: i+batch_size, :], target_tensor[:, i: i+batch_size, :],
                          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)

        print_loss_total += loss

    return print_loss_total

if __name__ == "__main__":
    path = "./data/chp_op18.mid"
    piano_data = midiToPianoroll(path, debug=True)
    print("shape of data ", piano_data.shape)
    time_len = 20
    output_len = 20
    hidden_dim = 256
    input_dim = 88
    output_dim = 88
    epoch = 200
    input_datax, input_datay = createSeqNetInputs([piano_data], time_len, output_len)
    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = DecoderRNN(input_dim, hidden_dim).to(device)
    for i in range(1, epoch+1):
        loss = trainIters(input_datax, input_datay, encoder1, decoder1, max_length=4000)
        print(f'{i} loss {loss}')
