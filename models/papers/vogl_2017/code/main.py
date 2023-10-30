import torch
import torch.nn as nn

class DrumRBM(nn.Module):
    # Restricted Boltzmann Machine for drum pattern generation
    def __init__(self):
        super(DrumRBM, self).__init__()
        self.n_visible = 64
        self.n_hidden = 500
        self.weight_matrix = nn.Parameter(torch.randn(self.n_hidden, self.n_visible) * 0.1)
        self.visible_bias = nn.Parameter(torch.zeros(self.n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(self.n_hidden))
        self.hidden_dropout = nn.Dropout(0.5)
        self.visible_dropout = nn.Dropout(0.1)

    def sample_hidden(self, visible):
        hidden_prob = torch.sigmoid(torch.matmul(visible, self.weight_matrix.t()) + self.hidden_bias)
        # hidden_prob = self.hidden_dropout(hidden_prob) #fails with dropout
        # print("hp",hidden_prob)
        return hidden_prob, torch.bernoulli(hidden_prob)
    
    def sample_visible(self, hidden):
        visible_prob = torch.sigmoid(torch.matmul(hidden, self.weight_matrix) + self.visible_bias)
        # visible_prob = self.visible_dropout(visible_prob) #fails with dropout
        # print("vp",visible_prob)
        return visible_prob, torch.bernoulli(visible_prob)
    