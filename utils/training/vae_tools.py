import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, beta=1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD

def vae_freebits_loss(recon_x, x, mu, logvar, beta=1, freebits=48):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.max(kl_loss, torch.tensor(freebits))
    return recon_loss + kl_loss

