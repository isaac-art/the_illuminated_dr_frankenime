import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.papers.warren_2022 import LatentDrummer

from utils import p_
from utils.training import make_deterministic

p_()
make_deterministic()
p_()

device = "cpu"

print(LatentDrummer)
p_()


# Fake data for testing
x = torch.randn(32, 1, 100, 100)
outs, mu, logvar = LatentDrummer(x)
# Should output torch.Size([32, 80]) torch.Size([32, 8])
assert outs[0].shape == torch.Size([32, 80])
assert outs[1].shape == torch.Size([32, 8])
print("Test passed")

# To test the sampling:
sample_z = torch.randn(32, 8)  # 32 samples in the 8-dimensional latent space
sample_outs = LatentDrummer.sample(sample_z)

 # Should output torch.Size([32, 80]) torch.Size([32, 8])
assert sample_outs[0].shape == torch.Size([32, 80])
assert sample_outs[1].shape == torch.Size([32, 8])

print("Sampling test passed")
