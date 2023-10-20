import torch
import torch.nn as nn

class LSTMEncoder(nn.Module)
    def __init__(self, input_dim, hidden_dim, bidi=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidi)
    
    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, bidi=False):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, bidirectional=bidi)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        prediction = self.softmax(prediction)
        return prediction, hidden, cell
    
class LSTMS2S(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(LSTMS2S, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
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