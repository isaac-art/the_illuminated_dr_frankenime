import torch
import torch.nn as nn

from models.core.lstms2s import LSTMEncoder, LSTMDecoder, LSMTS2STF

class ConvincingHarmony(nn.Module):
    def __init__(self, device="mps"):
        super(ConvincingHarmony, self).__init__()
        self.num_chorales = 405
        self.seq_len = 32
        self.input_dim = 130 # ?? During training, each sequence was split into shorter sequences of 32 timesteps.
        self.output_dim = 130 # 0-127 midi, 128 silence, 129 repeat
        self.hidden_dim = 256
        self.dropout = 0.2
        self.layers = 1
        self.device = device
        self.s2s = LSMTS2STF(self.input_dim, self.output_dim, self.layers, self.hidden_dim, self.dropout, self.device)

    def forward(self, x, y, teacher_force=0.5):
        return self.s2s(x, y, teacher_force=teacher_force)
    
    def generate(self, x):
        return self.s2s.generate(x)