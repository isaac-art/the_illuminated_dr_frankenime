import torch
import torch.nn as nn

class NegHalfOne(nn.Module):
    def forward(self, x):
        return 1.5 * torch.sigmoid(x) - 0.5

class LSTMVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layers):
        super(LSTMVAEEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, bidirectional=False, batch_first=True) # (seq_len, batch, vocab)
        # self.mu = nn.Linear(hidden_dim*2, latent_dim)
        # self.logvar = nn.Linear(hidden_dim*2, latent_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
    
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


class LSTMVAE(nn.Module):
    def __init__(self, seq_len, hidden_dim, latent_dim, layers, sig=False, negone=False):
        super(LSTMVAE, self).__init__()
        self.encoder = LSTMVAEEncoder(seq_len, hidden_dim, latent_dim, layers)
        self.decoder = LSTMVAEDecoder(latent_dim, hidden_dim, seq_len, layers, sig, negone)
    
    def forward(self, x):
        # print("forward vae", x.shape)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std
    

class GRUVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GRUVAEEncoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        _, h = self.rnn(x)
        h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class GRUVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(GRUVAEDecoder, self).__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        out, _ = self.rnn(z)
        out = self.fc(out)
        return out