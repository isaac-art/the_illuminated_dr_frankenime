import torch
import torch.nn as nn

from models.core import BaseModel, GRUVAEDecoder, GRUVAEEncoder

class MusicVAE(BaseModel):
    def __init__(self, input_dim, hidden_dim, latent_dim, beta):
        super(MusicVAE, self).__init__()
        self.encoder = GRUVAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GRUVAEDecoder(latent_dim, hidden_dim, input_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def sample(self):
        z = torch.randn(1, self.encoder.latent_dim)
        return self.decode(z)
    
    def sample_mu_logvar(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)
    
    def generate(self, z=None):
        if z is None:
            z = torch.randn(1, self.encoder.latent_dim)
        return self.decode(z)
