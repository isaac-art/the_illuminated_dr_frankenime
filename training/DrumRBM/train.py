import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.papers import DrumRBM

device = "mps"
drbm = DrumRBM().to(device)
optimizer = optim.SGD(drbm.parameters(), lr=0.001)
k = 1
latent_penalty = 0.1  # Penalty for latent selectivity
sparsity_penalty = 0.5  # Penalty for sparsity
n_samples = 1000
n_visible = 64

data = np.load('datasets/vogl_encodings.npy')
data = data.astype(np.float32)
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]


train_data_loader = DataLoader(TensorDataset(torch.from_numpy(train_data)), batch_size=256, shuffle=True)
test_data_loader = DataLoader(TensorDataset(torch.from_numpy(test_data)), batch_size=256, shuffle=True)

persistent_chain = torch.bernoulli(torch.from_numpy(train_data[:128])).to(device)

for epoch in range(2000):
    for i, (data,) in enumerate(train_data_loader):
        visible_data = data.to(device)
        _, h_data = drbm.sample_hidden(visible_data)
        for _ in range(k):
            _, h_sample = drbm.sample_hidden(persistent_chain)
            _, persistent_chain = drbm.sample_visible(h_sample)
            persistent_chain = persistent_chain.detach()
        _, h_fake = drbm.sample_hidden(persistent_chain)
        pos_phase = torch.matmul(visible_data.t(), h_data)
        neg_phase = torch.matmul(persistent_chain.t(), h_fake)

        loss = -torch.mean(pos_phase - neg_phase)
        latent_loss = torch.mean(torch.abs(torch.mean(h_data, dim=0) - torch.mean(h_fake, dim=0)))
        latent_loss = latent_penalty * latent_loss
        sparsity_loss = torch.mean(torch.abs(torch.mean(visible_data, dim=0) - torch.mean(persistent_chain, dim=0)))
        sparsity_loss = sparsity_penalty * sparsity_loss
        loss = loss - latent_loss - sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"E:{epoch}, I:{i}, Loss: {loss.item()}", end='\r')
    print(f"E:{epoch}, Loss: {loss.item()}")
# save
torch.save(drbm.state_dict(), f'archive/drbm_{epoch}.pt')