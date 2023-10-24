import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.general import p_
from utils.data import GillickDataMaker
from models.papers.gillick_2021 import WhatHowPlayVAE


# the Score and Groove inputs to this model are each 
# encoded (in this case with bidirectional LSTM encoders) 
# into separate latent variables Z1 and Z2 , which are both 
# independently trained to match standard normal distributions; 
# following Roberts et al.[32], we train using the free bits 
# method (hyper-parameters to balance the two loss terms in a VAE) 
# with a tolerance of 48 bits.

with open('datasets/gillick.pkl', 'rb') as f:
    dataset = pickle.load(f)
    # dataset = { 'combis': [], 'quantizeds': [], 'squasheds': [] }

gdm = GillickDataMaker()

# SPLIT DATA
combi_train = dataset['combis'][:int(len(dataset['combis'])*0.8)]
quantized_train = dataset['quantizeds'][:int(len(dataset['quantizeds'])*0.8)]
squashed_train = dataset['squasheds'][:int(len(dataset['squasheds'])*0.8)]
combi_test = dataset['combis'][int(len(dataset['combis'])*0.8):]
quantized_test = dataset['quantizeds'][int(len(dataset['quantizeds'])*0.8):]
squashed_test = dataset['squasheds'][int(len(dataset['squasheds'])*0.8):]

assert len(combi_train) == len(quantized_train) == len(squashed_train)
assert len(combi_test) == len(quantized_test) == len(squashed_test)
print("data loaded")
p_()

device = 'mps'
batch_size = 32
train_steps = 10000

model = WhatHowPlayVAE().to(device)
print(model)
p_()

optimizer = Adam(model.parameters(), lr=1e-3)
critereon = nn.CrossEntropyLoss()

# QUESTIONS 
# - velocities and notes are ints in midi range, 
#  but timing is float in seconds. 
# should we do something with the timings? if we do how do we get them back on decode??

# sequences are two-measures. but vary in length? does this matter. what do we do, pad out to longest?

print(len(combi_train), len(quantized_train), len(squashed_train)) # 18482 18482 18482

longest_combi = max(len(seq) for seq in combi_train)
longest_notes = max(len(seq) for seq in quantized_train)
longest_groove = max(len(seq) for seq in squashed_train)


# shoudl we use scalar?

# train groove
 
 
# train notes
# train combi


exit()
combis = torch.Tensor(combi_train)
quantizeds = torch.Tensor(quantized_train)
squasheds = torch.Tensor(squashed_train)

model.train()
for step in range(train_steps):
    # get batch
    batch_idx = torch.randint(0, len(combis), (batch_size,))
    combi_batch = combis[batch_idx].to(device)
    quantized_batch = quantizeds[batch_idx].to(device)
    squashed_batch = squasheds[batch_idx].to(device)
        
    # forward
    optimizer.zero_grad()
    combi_out = model.forward(drum=quantized_batch, rhythm=squashed_batch)
    quantized_out = model.forward_score(quantized_batch)
    squashed_out = model.forward_groove(squashed_batch)
    # loss
    loss = critereon(combi_out, combi_batch) + \
        critereon(quantized_out, quantized_batch) + \
        critereon(squashed_out, squashed_batch)
    # backward
    loss.backward()
    optimizer.step()
    # print
    print(f'step {step} loss {loss}', end='\r')
