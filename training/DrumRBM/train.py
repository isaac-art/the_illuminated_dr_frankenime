import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.papers.vogl_2017 import DrumRBM

drbm = DrumRBM() 
optimizer = optim.SGD(drbm.parameters(), lr=0.1)
k = 1
latent_penalty = 0.1  # Penalty for latent selectivity
sparsity_penalty = 0.5  # Penalty for sparsity
n_samples = 1000
n_visible = 64
synthetic_data = torch.bernoulli(torch.rand(n_samples, n_visible))
data_loader = DataLoader(TensorDataset(synthetic_data), batch_size=128, shuffle=True)
persistent_chain = torch.bernoulli(torch.rand(128, drbm.n_visible))  # Starting with random states


for epoch in range(200):  # Running for 10 epochs for demonstration
    for i, (data,) in enumerate(data_loader):
        visible_data = data
        
        # Positive Phase
        _, h_data = drbm.sample_hidden(visible_data)
        # Negative Phase
        for _ in range(k):
            _, h_sample = drbm.sample_hidden(persistent_chain.detach())
            _, persistent_chain = drbm.sample_visible(h_sample)
        # PCD Update
        _, h_fake = drbm.sample_hidden(persistent_chain)
        
        # Contrastive Divergence term
        loss = -torch.mean(torch.matmul(visible_data.t(), h_data) - torch.matmul(persistent_chain.t(), h_fake))
        
        # Latent Selectivity term
        latent_loss = torch.mean(torch.abs(torch.mean(h_data, dim=0) - torch.mean(h_fake, dim=0)))
        latent_loss = latent_penalty * latent_loss

        # Sparsity term
        sparsity_loss = torch.mean(torch.abs(torch.mean(visible_data, dim=0) - torch.mean(persistent_chain, dim=0)))
        sparsity_loss = sparsity_penalty * sparsity_loss

        # Total loss
        loss = loss - latent_loss - sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"E:{epoch}, I:{i}, Loss: {loss.item()}", end='\r')
    print(f"E:{epoch}, Loss: {loss.item()}")
# save
torch.save(drbm.state_dict(), f"../../../../weights/drbm.pt")


# Load the trained DRBM weights
drbm.load_state_dict(torch.load(f"../../../../weights/drbm.pt"))

# Initialize a visible unit with random values
v = torch.randn((1, 64))

# Gibbs sampling
for _ in range(k):
    # Sample from the hidden units given the visible units
    _, h = drbm.sample_hidden(v)
    # Sample from the visible units given the hidden units
    _, v = drbm.sample_visible(h)
print(v.int()) # final V is the sample