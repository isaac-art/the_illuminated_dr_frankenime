import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split


from utils.general import p_
from utils.data import GillickDataMaker
from utils.training import make_deterministic, vae_loss_bce, vae_loss_mse, vae_freebits_loss_bce, vae_freebits_loss_mse
from models.papers.gillick_2021 import WhatHowPlayAuxiliaryVAE, WhatHowPlayVAE

make_deterministic()
p_()

# the Score and Groove inputs to this model are each 
# encoded (in this case with bidirectional LSTM encoders) 
# into separate latent variables Z1 and Z2 , which are both 
# independently trained to match standard normal distributions; 
# following Roberts et al.[32], we train using the free bits 
# method (hyper-parameters to balance the two loss terms in a VAE) 
# with a tolerance of 48 bits.

notes = np.load('datasets/gillick_notes.npy') # (11627, 32, 9)  (measures, 16ths, drum(0,1))
timings = np.load('datasets/gillick_timings.npy') # (11627, 32, 18) (measures, 16ths, 9timings(0.0,1.0) 9offsets(-0.5,0.5))

notes_tensor = torch.tensor(notes, dtype=torch.float32)
timings_tensor = torch.tensor(timings, dtype=torch.float32)
dataset = TensorDataset(notes_tensor, timings_tensor)

total_samples = len(dataset)
train_samples = int(0.8 * total_samples)
val_samples = total_samples - train_samples
train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])

gdm = GillickDataMaker()
device = 'mps'
batch_size = 128*2
epochs = 10000
lr = 1e-4

# model = WhatHowPlayVAE(seq_len=9*2, hidden_dim=128, latent_dim=256, layers=1).to(device)
model = WhatHowPlayAuxiliaryVAE().to(device)

# print(model)
optimizer = Adam(model.parameters(), lr=lr)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# TRAIN
# Training Loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch_idx, (drum, rhythm) in enumerate(train_loader):  # replace with your DataLoader
        # print("-"*120)
        # print("batch_idx", batch_idx)
        optimizer.zero_grad()
        drum = drum.to(device) # (batch, 32, 9)
        rhythm = rhythm.to(device) # (batch, 32, 18)

        assert not torch.isnan(drum).any(), "Drum data contains NaN"
        assert not torch.isnan(rhythm).any(), "Rhythm data contains NaN"

        joint = torch.cat((drum, rhythm), dim=2) # (batch, 32, 27)
        joint = joint.to(device)
        
        recon_drum, z1, dmu, dlogvar = model.score_vae(drum)
        # print("recon_drum", recon_drum.cpu().detach().numpy())

        recon_rhythm, z2, rmu, rlogvar = model.groove_vae(rhythm)

        joint_z = torch.cat((z1, z2), dim=2)
        joint_recon = model.joint_decoder(joint_z)

        dloss = vae_freebits_loss_mse(recon_drum, drum, dmu, dlogvar)
        rloss = vae_freebits_loss_mse(recon_rhythm, rhythm, rmu, rlogvar)
        # jloss = F.mse_loss(joint_recon, joint, reduction='sum')
        jloss_d = vae_freebits_loss_mse(joint_recon, joint, dmu, dlogvar)
        jloss_r = vae_freebits_loss_mse(joint_recon, joint, rmu, rlogvar)

        loss = rloss + dloss + jloss_d + jloss_r
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        train_loss += loss.item()
        optimizer.step()
        num_batches += 1
        avg_loss = train_loss / num_batches
        print(f'Epoch: {epoch}, Batch: {batch_idx+1}/{len(train_loader)}, Average Loss: {avg_loss}', end='\r')
    print(f'Epoch: {epoch}, Batch: {batch_idx+1}/{len(train_loader)}, Average Loss: {avg_loss}')
    # save epoch
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f'weights/whathowplayauxvae_{epoch}_{avg_loss}.pt')
    
torch.save(model.state_dict(), f'weights/whathowplayauxvae.pt')
