import torch
import torch.nn as nn

from models.core import BaseModel, ConvVAE, ConvEncoder, ConvDecoder

class LatentDrummer(BaseModel):
    def __init__(self):
        super(LatentDrummer, self).__init__()
        self.enc_config = [(1, 8, 10, 5), (8, 8, 4, 2), (8, 4, 4, 1)]
        self.latent_dim = 8
        self.dec1_config = [(8, 5 * 16)]
        self.dec2_config = [(8, 8)]
        LatentDrummerEncoder = ConvEncoder(self.enc_config) #ROLI LIGHTPAD DRAWINGS 100x100 binary image input
        LatentDrummerDecoder1 = ConvDecoder(self.dec1_config) # velocities out 5*16 matrix
        LatentDrummerDecoder2 = ConvDecoder(self.dec2_config) # Markov State

        LatentDrummer = ConvVAE(LatentDrummerEncoder, self.latent_dim, [LatentDrummerDecoder1, LatentDrummerDecoder2])

    def forward(self, x):
        output, mu, logvar = self.LatentDrummer(x)
        return output, mu, logvar
    
    def sample(self):
        z = torch.randn(1, self.latent_dim)
        return self.LatentDrummer.decode(z)
    
    def generate(self, z):
        return self.LatentDrummer.decode(z)
    