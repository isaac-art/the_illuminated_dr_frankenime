import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layers):
        super(LSTMVAEEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, bidirectional=True) # (seq_len, batch, vocab)
        self.mu = nn.Linear(hidden_dim*2, latent_dim)
        self.logvar = nn.Linear(hidden_dim*2, latent_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[-1] # last hidden state
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class LSTMVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, layers):
        super(LSTMVAEDecoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, layers, bidirectional=True)
        self.dense = nn.Linear(hidden_dim*2, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.dense(x[-1])
    

class WhatHowPlayVAE(nn.Module):
    def __init__(self, seq_len=300, hidden_dim = 128, latent_dim = 16, layers = 1):
        super(WhatHowPlayVAE, self).__init__()
        self.encoder = LSTMVAEEncoder(seq_len, hidden_dim, latent_dim, layers)
        self.decoder = LSTMVAEDecoder(latent_dim, hidden_dim, seq_len, layers)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

class WhatHowPlayAuxiliaryVAE(nn.Module):
    def __init__(self, seq_len=300):
        super(WhatHowPlayAuxiliaryVAE, self).__init__()
        hidden_dim = 128
        latent_dim = 16
        layers = 1
        self.score_vae = WhatHowPlayVAE(seq_len, hidden_dim, latent_dim, layers)
        self.groove_vae = WhatHowPlayVAE(seq_len, hidden_dim, latent_dim, layers)
        self.joint_decoder = LSTMVAEDecoder(latent_dim*2, hidden_dim, seq_len, layers)
    
    def forward(self, drum, rhythm): #jointforward
        dmu, dlogvar = self.score_vae.encoder(drum)
        rmu, rlogvar = self.groove_vae.encoder(rhythm)
        z1 = self.reparameterize(dmu, dlogvar)
        z2 = self.reparameterize(rmu, rlogvar)
        z = torch.cat((z1, z2), dim=1)
        return self.joint_decoder(z)

    def forward_score(self, x):
        return self.score_vae(x)

    def forward_groove(self, x):
        return self.groove_vae(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def sample(self, z1, z2):
        z = torch.cat((z1, z2), dim=1)
        return self.joint_decoder(z)
    
    def sample_score(self, z): 
        return self.score_vae.decoder(z)
    
    def sample_groove(self, z): 
        return self.groove_vae.decoder(z)

