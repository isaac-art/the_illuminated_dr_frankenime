
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder

class ScoreTransformer(nn.Module):
    def __init__(self, num_tokens=10000, max_seq_len=1024, 
            d_model=512, depth=12, nhead=8, rel_pos_bias=True):
        super(ScoreTransformer, self).__init__()
        self.model = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                dim = d_model,
                depth = depth,
                heads = nhead,
                rel_pos_bias = rel_pos_bias 
            )
        )
        
    def forward(self, x):
        return self.model(x)
