import torch.nn as nn
from utils.errors import *

class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BaseModel, self).__init__(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        # The forward pass, used for training
        raise NotImplementedError("This method should be overridden by subclass")

    def encode(self, *args, **kwargs):
        # Encoder, for VAE, etc.
        raise NotImplementedError("This method should be overridden by subclass")

    def decode(self, *args, **kwargs):
        # Decoder, for VAE, etc.
        raise NotImplementedError("This method should be overridden by subclass")

    def sample(self):
        # Sample from latent space, for VAE, etc. 
        # this takes no arguments because it should be random/from a distribution
        raise NotImplementedError("This method should be overridden by subclass")

    def generate(self, *args, **kwargs):
        # main generate function, for all models
        # this takes arguments if needed
        raise NotImplementedError("This method should be overridden by subclass")