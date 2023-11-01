import torch
import torch.nn as nn

from models.core import Seq2SeqBiLSTM, BaseModel

class ConvincingHarmony(BaseModel):
    def __init__(self):
        super(ConvincingHarmony, self).__init__()
        self.vocab_size = 130
        self.embed_dim = 32 
        self.hidden_dim = 256
        self.dropout = 0.2
        self.layers = 1
        self.s2s = Seq2SeqBiLSTM(vocab_size=self.vocab_size, embed_size=self.embed_dim, hidden_size=self.hidden_dim)

    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        return self.s2s(x, y, teacher_forcing_ratio=teacher_forcing_ratio)
