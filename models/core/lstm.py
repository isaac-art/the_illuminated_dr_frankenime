import torch
import torch.nn as nn
from typing import Tuple, Optional

class LSTMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, input_dim)
        # self.dropout = nn.Dropout(0.5) #we apply dropout of 0.5 on all the layers, excluding the recurrent connection


    def forward(self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rnn_out, (h, c) = self.rnn(x, h)
        return rnn_out, h, c
        # token_pred = self.fc(rnn_out)
        # token_pred = self.dropout(token_pred)
        # return rnn_out, h, c, token_pred