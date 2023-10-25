import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.general import p_
from utils.data import GillickDataMaker
from utils.training import vae_freebits_loss
from models.papers.gillick_2021 import WhatHowPlayAuxiliaryVAE


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

model = WhatHowPlayAuxiliaryVAE().to(device)
print(model)
p_()

optimizer = Adam(model.parameters(), lr=1e-3)
critereon = nn.CrossEntropyLoss()

# QUESTIONS 
# - velocities and notes are ints in midi range, 
# sequences are two-measures. but vary in length? does this matter. what do we do, pad out to longest?

# vis distribution of values in a random offset for each set
rng = np.random.randint(0, len(combi_train))
combi_sample = combi_train[rng]
quantized_sample = quantized_train[rng]
squashed_sample = squashed_train[rng]

plt.hist(combi_sample, bins=100)
plt.title("combi")
plt.show()
plt.hist(quantized_sample, bins=100)
plt.title("quantized")
plt.show()
plt.hist(squashed_sample, bins=100)
plt.title("squashed")
plt.show()


print(len(combi_train), len(quantized_train), len(squashed_train)) # 8527 8527 8527

longest_combi = max(len(seq) for seq in combi_train)
longest_notes = max(len(seq) for seq in quantized_train)
longest_groove = max(len(seq) for seq in squashed_train)

print(longest_combi, longest_notes, longest_groove) # 436 109 327
longest = max(longest_combi, longest_notes, longest_groove) # 436
# pad out all seqs to longest
combi_train = [seq + [0]*(longest-len(seq)) for seq in combi_train]
quantized_train = [seq + [0]*(longest-len(seq)) for seq in quantized_train]
squashed_train = [seq + [0]*(longest-len(seq)) for seq in squashed_train]

for i in range(5):
    rng = np.random.randint(0, len(combi_train))
    print(combi_train[rng][:5], quantized_train[rng][:5], squashed_train[rng][:5])


exit()
def normalize_to_tensor(data):
    data = torch.Tensor(data)
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    feature_means = torch.mean(data, dim=0)
    feature_stds = torch.std(data, dim=0)
    print("Feature means:", feature_means[:5])
    print("Feature std devs:", feature_stds[:5])

    std[std == 0] = 1.0
    data_normalized = (data - mean) / std
    return data_normalized

combi_train = normalize_to_tensor(combi_train)
quantized_train = normalize_to_tensor(quantized_train)
squashed_train = normalize_to_tensor(squashed_train)

# print min max of each
print(torch.min(combi_train), torch.max(combi_train)) # -4.9533 92.33
print(torch.min(quantized_train), torch.max(quantized_train)) # -4.9533 92.33
print(torch.min(squashed_train), torch.max(squashed_train)) # -2.6482 92.3309


# train groove vae
# train score vae
# train combi vae



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
