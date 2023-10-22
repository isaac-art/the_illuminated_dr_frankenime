import torch
import torch.nn as nn

from models.core.conv_vae import ConvVAE, ConvEncoder, ConvDecoder

enc_config = [(1, 8, 10, 5), (8, 8, 4, 2), (8, 4, 4, 1)]
latent_dim = 8
dec1_config = [(8, 5 * 16)]
dec2_config = [(8, 8)]

LatentDrummerEncoder = ConvEncoder(enc_config) #ROLI LIGHTPAD DRAWINGS 100x100 binary image input
LatentDrummerDecoder1 = ConvDecoder(dec1_config) # velocities out 5*16 matrix
LatentDrummerDecoder2 = ConvDecoder(dec2_config) # Markov State

LatentDrummer = ConvVAE(LatentDrummerEncoder, latent_dim, [LatentDrummerDecoder1, LatentDrummerDecoder2])
