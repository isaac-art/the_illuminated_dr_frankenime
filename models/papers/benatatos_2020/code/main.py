import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core import BaseModel, LSTMLayer, StackMemory

class BachDuet(BaseModel):
    def __init__(self, token_dim=135, cpc=12, rhythm=8, hidden_dim=400, stack_size=32, key_hidden_dim=300, num_embedding_dim=50):
        super(BachDuet, self).__init__()
        self.embedding_midi = nn.Embedding(token_dim, num_embedding_dim)
        self.embedding_cpc = nn.Embedding(cpc, num_embedding_dim)
        self.embedding_rhythm = nn.Embedding(rhythm, num_embedding_dim)
        self.note_rnn = LSTMLayer(3 * num_embedding_dim, hidden_dim, num_layers=2)
        self.stack_memory = StackMemory(hidden_dim, stack_size)
        self.key_rnn = LSTMLayer(hidden_dim+cpc, key_hidden_dim, num_layers=1)
        self.prev_key = torch.zeros(1, cpc) # we initialize the previous key to 0

    def forward(self, xr, xcpc, xma):
        device = xr.device
        self.prev_key = self.prev_key.to(device)

        x_midi = self.embedding_midi(xma)
        x_cpc = self.embedding_cpc(xcpc)
        x_rhythm = self.embedding_rhythm(xr)
        print("embeddings", x_midi.shape, x_cpc.shape, x_rhythm.shape) #([32, 256, 50])([32, 256, 50])([32, 256, 50])
        x_embedded = torch.cat((x_midi, x_cpc, x_rhythm), dim=2)
        print("x_embedded", x_embedded.shape) #([32, 256, 150]

        rnn_out, rh, rc, rtoken_pred = self.note_rnn(x_embedded)
        print("rtoken_pred", rtoken_pred.shape) ([32, 256, 150])

        #The input to the key LSTM, is a combination of the previous key, 
        # with the hidden state of the note LSTM unit.
        # join rh[-1] and self.prev_key from [32,400] and [1,12] to 32,412
        combined_context = torch.cat((rh[-1], self.prev_key), dim=1)
        print("cc",combined_context.shape) ([32, 412])
        krnn_out, kh, kc, ktoken_pred = self.key_rnn(combined_context)
        print("kp",ktoken_pred.shape) ([32, 412])
        self.prev_key = ktoken_pred
        return rtoken_pred, ktoken_pred
