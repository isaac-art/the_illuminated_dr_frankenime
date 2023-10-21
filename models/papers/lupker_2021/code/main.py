import torch
import torch.nn as nn

from models.core.encodings import RelativePositionalEncoding, PositionalEncoding

class ScoreTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=8, dim_feedforward=2047, vocab_size=245, max_len=2047, dropout=0.1):
        super(ScoreTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.positional_encoding = RelativePositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory, tgt_mask=None):
        # print("x", x.shape) # torch.Size([32, 2047])
        x = self.embedding(x)  # shape: [batch_size, seq_len, d_model]
        # print("x embedded", x.shape) # torch.Size([32, 2047, 512])
        x = x.permute(1, 0, 2)  # shape: [seq_len, batch_size, d_model]
        # print("x permuted", x.shape) # torch.Size([2047, 32, 512])
        x = self.positional_encoding(x) #torch.Size([2047, 32, 512])
        # print("x pos encoded", x.shape)
        # print("memory", memory.shape)
        # print("tgt_mask", tgt_mask.shape)
        output = self.decoder(x, memory, tgt_mask)
        # print("decoded", output.shape)
        output = self.output_layer(output.permute(1, 0, 2))  # shape: [batch_size, seq_len, vocab_size]
        # print("output", output.shape)
        return output