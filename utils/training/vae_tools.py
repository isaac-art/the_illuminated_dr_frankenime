import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss_mse(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    epsilon = 1e-10
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar + epsilon).exp())
    return MSE + KLD

def vae_freebits_loss_mse(recon_x, x, mu, logvar, freebits=48):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.max(kl_loss, torch.tensor(freebits))
    return recon_loss + kl_loss

def vae_loss_bce(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    epsilon = 1e-10
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar + epsilon).exp())
    return BCE + KLD

def vae_freebits_loss_bce(recon_x, x, mu, logvar, freebits=48):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.max(kl_loss, torch.tensor(freebits))
    return recon_loss + kl_loss