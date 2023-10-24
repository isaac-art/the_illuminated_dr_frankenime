import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim

from utils import p_
from utils.training import make_deterministic
from models.papers.faitas_2019 import ConvincingHarmony

p_()
make_deterministic()

device = "mps"
num_chorales = 405
seq_len = 32
parts = 4 # (soprano, alto), (tenor, bass) pairs
vocab_size = 130# 0-127 MIDI notes + 128 for silence + 129 for repeat
n_epochs = 248
batch_size = 128
lr = 0.001

# LOAD DATA
# [seq_len, batch_size, vocab_size] [32, 128, 130]

# LOAD MODEL

model = ConvincingHarmony(device).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# BEGIN TRAINING
x = torch.rand(seq_len, batch_size, vocab_size)
y = torch.rand(seq_len, batch_size, vocab_size)
print(x.shape, y.shape) #torch.Size([32, 128, 130]) #seq,batch,feature
x = x.to(device)
y = y.to(device)
output = model(x, y)
