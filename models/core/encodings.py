import torch
import math
import torch.nn as nn

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(RelativePositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device)
        positions = positions.view(-1, 1).expand(-1, x.size(0)).contiguous().view(-1)
        pos_enc = self.encoding.index_select(0, positions).view(x.size(1), x.size(0), -1).permute(1, 0, 2)
        return pos_enc.to(x.device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)