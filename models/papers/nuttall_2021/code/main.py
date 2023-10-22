import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder

class RhythmTransformerXL(nn.Module):
    def __init__(self, num_tokens=40, max_seq_len=64, max_mem_len=128,
            d_model=512, depth=6, nhead=8):
        super(RhythmTransformerXL, self).__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.d_model = d_model
        self.depth = depth
        self.nhead = nhead

        self.model_xl = TransformerWrapper(
            num_tokens = self.num_tokens,
            max_seq_len = self.max_seq_len,
            max_mem_len = self.max_mem_len,
            attn_layers = Decoder(
                dim = self.d_model,
                depth = self.depth,
                heads = self.nhead,
                rel_pos_bias = True, # recurrence requires relative positional bias
            )
        )

    def forward(self, x, mems=False, return_mems=True):
        if mems: return self.model_xl(x, mems=mems, return_mems=return_mems)
        else: return self.model_xl(x, return_mems=return_mems)