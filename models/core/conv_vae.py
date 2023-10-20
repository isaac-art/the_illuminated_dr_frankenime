
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, conv_layers):
        super(ConvEncoder, self).__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(*params) for params in conv_layers])
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.conv_layers:
            x = self.activation(layer(x))
        x = torch.flatten(x, 1)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, layers):
        super(ConvDecoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(*params) for params in layers])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvVAE(nn.Module):
    def __init__(self, encoder, latent_dim, decoders):
        super(ConvVAE, self).__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.decoders = nn.ModuleList(decoders)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:2*self.latent_dim]
        z = self.reparameterize(mu, logvar)
        outs = [decoder(z) for decoder in self.decoders]
        return outs, mu, logvar
    
    def sample(self, z):
        outs = [decoder(z) for decoder in self.decoders]
        return outs

def vae_loss(recon_x1, x1, recon_x2, x2, mu, logvar):
    recon_loss1 = nn.MSELoss()(recon_x1, x1)
    recon_loss2 = nn.MSELoss()(recon_x2, x2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_recon_loss = recon_loss1 + recon_loss2
    return total_recon_loss + kl_div

if __name__ == "__main__":
    # Encoder configuration
    enc_config = [(1, 8, 10, 5), (8, 8, 4, 2), (8, 4, 4, 1)]
    # latent
    latent_dim = 8
    # Decoder configurations
    dec1_config = [(8, 5 * 16)]
    dec2_config = [(8, 8)]
    
    encoder = ConvEncoder(enc_config)
    decoder1 = ConvDecoder(dec1_config)
    decoder2 = ConvDecoder(dec2_config)

    vae_model = ConvVAE(encoder, latent_dim, [decoder1, decoder2])

    # Fake data for testing
    x = torch.randn(32, 1, 100, 100)  # 32 samples of 100x100 binary images
    outs, mu, logvar = vae_model(x)
    # Should output torch.Size([32, 80]) torch.Size([32, 8])
    assert outs[0].shape == torch.Size([32, 80])
    assert outs[1].shape == torch.Size([32, 8])
    print("Test passed")

     # To test the sampling:
    sample_z = torch.randn(32, 8)  # 32 samples in the 8-dimensional latent space
    sample_outs = vae_model.sample(sample_z)

     # Should output torch.Size([32, 80]) torch.Size([32, 8])
    assert sample_outs[0].shape == torch.Size([32, 80])
    assert sample_outs[1].shape == torch.Size([32, 8])
    
    print("Sampling test passed")