import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder

class RhythmTransformerXL(nn.Module):
    def __init__(self, num_tokens=10000, max_seq_len=2048, max_mem_len=256,
            d_model=512, depth=6, nhead=8, rel_pos_bias=False):
        super(RhythmTransformerXL, self).__init__()
        self.model_xl = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            max_mem_len = max_mem_len,
            attn_layers = Decoder(
                dim = d_model,
                depth = depth,
                heads = nhead,
                rel_pos_bias = rel_pos_bias
            )
        )

    def forward(self, x):
        return self.model(x)
