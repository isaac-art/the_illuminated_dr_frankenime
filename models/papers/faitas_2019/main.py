import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=False):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional)
        
    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1) 
        
    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        prediction = self.softmax(prediction)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
            
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        enc_out, hidden, cell = self.encoder(src)
        
        if self.encoder.lstm.bidirectional:
            num_layers = hidden.shape[0] // 2
            hidden = torch.cat([hidden[:num_layers], hidden[num_layers:]], dim=2)
            cell = torch.cat([cell[:num_layers], cell[num_layers:]], dim=2)
        
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs




device = torch.device('mps')
NUM_CHORALES = 405  # Number of chorales in the dataset
SEQ_LEN = 256  # Length of each sequence
NOTES = 130  # 0-127 MIDI notes + 128 for silence + 129 for repeat
BATCH_SIZE = 128  # Batch size for training
PARTS = 4  # soprano, alto, tenor, bass

INPUT_DIM = 256
OUTPUT_DIM = 256
HIDDEN_DIM = 256
BIDIRECTIONAL = True  # Set False for unidirectional
BATCH_SIZE = 128
N_EPOCHS = 400 if not BIDIRECTIONAL else 248  # Set accordingly
LR = 0.001
DROPOUT = 0.2
teacher_forcing_ratio = 0.5  # 50% of the time

def apply_teacher_forcing(decoder_output, target_seq):
    return target_seq if np.random.random() < teacher_forcing_ratio else decoder_output


# Initialize models and optimizer
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, BIDIRECTIONAL)
decoder = Decoder(OUTPUT_DIM, HIDDEN_DIM * 2 if BIDIRECTIONAL else HIDDEN_DIM)
model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Loss function
criterion = nn.CrossEntropyLoss()

# fake data for testing
soprano = np.random.randint(0, NOTES, (NUM_CHORALES * 2, SEQ_LEN))
alto = np.random.randint(0, NOTES, (NUM_CHORALES * 2, SEQ_LEN))
tenor = np.random.randint(0, NOTES, (NUM_CHORALES * 2, SEQ_LEN))
bass = np.random.randint(0, NOTES, (NUM_CHORALES * 2, SEQ_LEN))

input_train_1, target_train_1 = soprano, alto
input_train_2, target_train_2 = tenor, bass

