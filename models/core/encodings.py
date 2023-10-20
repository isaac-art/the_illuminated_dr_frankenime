import torch
import math
import torch.nn as nn

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # Add a dimension for the batch size
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, x):
        seq_len = x.size(1)
        pos = self.pos_enc[:seq_len, :, :]
        return pos
