import torch.nn as nn

class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.5) #we apply dropout of 0.5 on all the layers, excluding the recurrent connection

    def forward(self, x):
        rnn_out, (h, c) = self.rnn(x)
        token_pred = self.fc(rnn_out)
        token_pred = self.dropout(token_pred)
        return rnn_out, h, c, token_pred
    