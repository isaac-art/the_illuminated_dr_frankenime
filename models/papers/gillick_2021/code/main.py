import torch
import torch.nn as nn
import torch.nn.functional as F

class NegHalfOne(nn.Module):
    def forward(self, x):
        return 1.5 * torch.sigmoid(x) - 0.5

class LSTMVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layers):
        super(LSTMVAEEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, bidirectional=True, batch_first=True) # (seq_len, batch, vocab)
        self.mu = nn.Linear(hidden_dim*2, latent_dim)
        self.logvar = nn.Linear(hidden_dim*2, latent_dim)
    
    def forward(self, x):
        # print("forward encoder", x.shape)
        x, _ = self.lstm(x)
        # print("forward vae encoder", x.shape) #(batch, seq_len, hidden_dim*2) 32, 32, 1024
        # x = x[-1] # last hidden state
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class LSTMVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, layers, sig=False, negone=False):
        super(LSTMVAEDecoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, layers, bidirectional=False, batch_first=True)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.sig = sig
        self.negone = negone
        if sig: self.sigmoid = nn.Sigmoid()
        if negone: self.neghalfone = NegHalfOne()
    
    def forward(self, x):
        # print("forward decoder", x.shape)
        x, _ = self.lstm(x)
        # print("forward vae decoder", x.shape) # (batch, seq_len, hidden_dim) 32, 32, 512
        x = self.dense(x)
        if self.sig: return self.sigmoid(x)
        if self.negone: return self.neghalfone(x)
        return(x)


class WhatHowPlayVAE(nn.Module):
    def __init__(self, seq_len, hidden_dim, latent_dim, layers, sig=False, negone=False):
        super(WhatHowPlayVAE, self).__init__()
        self.encoder = LSTMVAEEncoder(seq_len, hidden_dim, latent_dim, layers)
        self.decoder = LSTMVAEDecoder(latent_dim, hidden_dim, seq_len, layers, sig, negone)
    
    def forward(self, x):
        # print("forward vae", x.shape)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std
    

class WhatHowPlayAuxiliaryVAE(nn.Module):
    def __init__(self, device="mps"):
        super(WhatHowPlayAuxiliaryVAE, self).__init__()
        self.device = device
        seq_len = 9
        hidden_dim = 512
        latent_dim = 256
        layers = 1
        self.score_vae = WhatHowPlayVAE(seq_len, hidden_dim, latent_dim, layers, sig=True)
        self.groove_vae = WhatHowPlayVAE(seq_len*2, hidden_dim, latent_dim, layers, negone=True)
        self.joint_decoder = LSTMVAEDecoder(latent_dim*2, hidden_dim, seq_len*3, layers)
    
    def forward(self, drum, rhythm): #jointforward
        # print("forward joint", drum.shape, rhythm.shape)
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

