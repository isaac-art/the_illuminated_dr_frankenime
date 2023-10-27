import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core import LSTMVAEDecoder, LSTMVAE
    
class WhatHowPlayAuxiliaryVAE(nn.Module):
    def __init__(self, device="mps"):
        super(WhatHowPlayAuxiliaryVAE, self).__init__()
        self.device = device
        seq_len = 9
        hidden_dim = 512
        latent_dim = 256
        layers = 1
        self.score_vae = LSTMVAE(seq_len, hidden_dim, latent_dim, layers, sig=True)
        self.groove_vae = LSTMVAE(seq_len*2, hidden_dim, latent_dim, layers, negone=True)
        self.joint_decoder = LSTMVAEDecoder(latent_dim*2, hidden_dim, seq_len*3, layers)
    
    def forward(self, drum, rhythm): #jointforward
        # print("forward joint", drum.shape, rhythm.shape)
        dmu, dlogvar = self.score_vae.encoder(drum)
        rmu, rlogvar = self.groove_vae.encoder(rhythm)
        z1 = self.reparameterize(dmu, dlogvar)
        z2 = self.reparameterize(rmu, rlogvar)
        z = torch.cat((z1, z2), dim=1)
        return self.joint_decoder(z), z1, z2, dmu, dlogvar, rmu, rlogvar

    def forward_score(self, x):
        return self.score_vae(x)

    def forward_groove(self, x):
        return self.groove_vae(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def sample(self):
        z1 = torch.randn(32, 256).to(self.device)
        z2 = torch.randn(32, 256).to(self.device)
        z = torch.cat((z1, z2), dim=1)
        return self.joint_decoder(z)
    
    def sample_score(self): 
        z = torch.randn(32, 256).to(self.device)
        return self.score_vae.decoder(z)
    
    def sample_groove(self): 
        z = torch.randn(32, 256).to(self.device)
        return self.groove_vae.decoder(z)

