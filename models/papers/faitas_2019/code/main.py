import torch
import torch.nn as nn

from models.core.lstms2s import LSTMEncoder, LSTMDecoder, LSTMS2S

class ConvincingHarmony(nn.Module):
    def __init__(self, bidi=True):
        super(ConvincingHarmony, self).__init__()
        self.num_chorales = 405
        self.seq_len = 256
        self.notes = 130 #0-127 midi, 128 silence, 129 repeat
        self.parts = 4 #soprano, alto, tenor, bass
        self.input_dim = 256
        self.output_dim = 256
        self.hidden_dim = 256
        self.bidi = bidi
        pass

    def forward(self, x):
        return x
