import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, layers, hidden_dim, dropout, bidi):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, bidirectional=bidi)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.dropout(outputs)
        return outputs, hidden, cell 

    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, layers, hidden_dim, dropout, bidi):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        if bidi: inp = hidden_dim*2
        else: inp = hidden_dim
        self.lstm = nn.LSTM(inp, hidden_dim, layers, bidirectional=bidi)
        self.fc_out = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sm_out = nn.Softmax(output_dim)

    def forward(self, input, hidden, cell):
        print("Decoder forward")
        # input -> [1, batchsize, output(vocab)]->[1, 32, 130]
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # hc -> [2, 32(batch), 256]
        output = self.dropout(output)
        prediction = self.fc_out(output.squeeze(0))
        prediction = self.sm_out(prediction) #softmax
        return prediction, hidden, cell # p-> [1, n, 130]
    
class LSMTS2STF(nn.Module):
    def __init__(self, input_dim, output_dim, layers, hidden_dim, dropout, bidi=False, device="mps"):
        super(LSMTS2STF, self).__init__()
        # LSTM Sequence to Sequence model with teacher forcing
        # if you dont want forcing just set teacher_force to 0
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.bidi=bidi
        self.encoder = LSTMEncoder(self.input_dim, self.layers, self.hidden_dim, self.dropout, self.bidi)
        self.decoder = LSTMDecoder(self.output_dim, self.layers, self.hidden_dim, self.dropout, self.bidi)
    
    def forward(self, x, y, teacher_force=0.5):
        print(x.shape, y.shape) #torch.Size([32, 128, 130]) torch.Size([32, 128, 130])
        o, h, c = self.encoder(x)
        print("encoded", o.shape, h.shape, c.shape) #torch.Size([32, 128, 512]) torch.Size([2, 128, 256]) torch.Size([2, 128, 256])
        y_len = y.shape[0] #seq len # y -> (seqlen, batch, outdim)
        batch_size = y.shape[1] #batch 
        vocab_size = self.decoder.output_dim 
        print("v_size", vocab_size) # 130
        # concat h and c 
        if self.bidi:
            print("is bidi, cat'in")
            h = torch.cat((h[0], h[1]), dim=2)
            c = torch.cat((c[0], c[1]), dim=2)
        outs = torch.zeros(y_len, batch_size, vocab_size).to(self.device)
        inp = y[0,:]
        for t in range(1, y_len):
            print(t,"/",y_len)
            o, h, c = self.decoder(inp, h, c)
            print("decoded", t)
            outs[t] = o
            use_the_force = torch.rand(1).item() > teacher_force
            top = o.argmax(1) # highest predicted token
            print(top)
            inp = y[t] if use_the_force else top
        return outs

    def generate(self, x):
        max_len = x.shape[0]
        batch_size = x.shape[1]
        outs = torch.zeros(max_len, batch_size, self.output_dim).to(self.device)
        inp = x[0,:]
        enc_outs, h, c = self.encoder(x)
        for t in range(1, max_len):
            o, h, c = self.decoder(inp, h, c)
            outs[t] = o
            top = o.argmax(1)
            inp = top
        return outs