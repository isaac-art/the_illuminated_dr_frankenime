import torch
import torch.nn as nn

from models.core.encodings import RelativePositionalEncoding

class ScoreTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, vocab_size, max_len=2048):
        super(ScoreTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = RelativePositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, memory, tgt_mask=None):
        x = self.embedding(x)
        x += self.positional_encoding(x)
        output = self.decoder(x, memory, tgt_mask)
        output = self.output_layer(output)
        return output
