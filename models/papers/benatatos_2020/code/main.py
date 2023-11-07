import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core import BaseModel, LSTMLayer, StackMemory

class BachDuet(BaseModel):
    def __init__(self, token_dim=135, cpc=24, rhythm=8, hidden_dim=400, stack_size=32, key_hidden_dim=300, num_embedding_dim=50):
        super(BachDuet, self).__init__()
        self.embedding_midi = nn.Embedding(token_dim, num_embedding_dim)
        self.embedding_cpc = nn.Embedding(cpc, num_embedding_dim)
        self.embedding_rhythm = nn.Embedding(rhythm, num_embedding_dim)
        self.note_rnn = nn.LSTM(input_size=num_embedding_dim*3, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        # self.note_rnn = LSTMLayer(3 * num_embedding_dim, hidden_dim, num_layers=2)
        # self.stack_memory = StackMemory(hidden_dim, stack_size) # This doesnt make sense to me?
        # self.key_rnn = LSTMLayer(hidden_dim+cpc, key_hidden_dim, num_layers=1)
        self.key_rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        # self.prev_key = torch.zeros(1, cpc) # we initialize the previous key to 0
        self.fc = nn.Linear(hidden_dim, token_dim)
        self.fckey = nn.Linear(hidden_dim, cpc)
        self.dropout = nn.Dropout(0.5) #we apply dropout of 0.5 on all the layers, excluding the recurrent connection

    def forward(self, xr, xcpc, xma):
        # rhythm, cpc, midi
        # device = xr.device
        # self.prev_key = self.prev_key.to(device)
        x_midi = self.embedding_midi(xma)
        x_cpc = self.embedding_cpc(xcpc)
        x_rhythm = self.embedding_rhythm(xr)
        # print("embeddings", x_midi.shape, x_cpc.shape, x_rhythm.shape) #([32, 256, 50])([32, 256, 50])([32, 256, 50])
        x_embedded = torch.cat((x_midi, x_cpc, x_rhythm), dim=2)
        # print("x_embedded", x_embedded.shape) #([32, 256, 150]
        rnn_out, hidden = self.note_rnn(x_embedded) #([32, 256, 400])
        token_pred = self.fc(rnn_out) #([32, 256, 135])
        # token_pred = self.dropout(token_pred) #([32, 256, 135])   
        token_pred_sm = torch.softmax(token_pred, dim=1)
        krnn_out, khidden = self.key_rnn(rnn_out) #([32, 256, 400])
        key_pred = self.fckey(krnn_out) #([32, 256, 24])
        # key_pred = self.dropout(key_pred) #([32, 256, 24])
        key_pred_sm = torch.softmax(key_pred, dim=1)
        return token_pred, token_pred_sm, key_pred, key_pred_sm

        # hidden_last = rh[-1].unsqueeze(1)
        # # print("hidden_last", hidden_last.shape) ([32, 1, 400])
        
        # #The input to the key LSTM, is a combination of the previous key, 
        # # with the hidden state of the note LSTM unit.
        # # join rh last and self.prev_key from [32,1, 400] and [1,24] to [32, 1, 424]

        # combined_context = torch.cat((hidden_last, self.prev_key.unsqueeze(1)), dim=2)
        # print("cc",combined_context.shape) ([32, 412])
        # krnn_out, kh, kc, ktoken_pred = self.key_rnn(combined_context)
        # print("kp",ktoken_pred.shape) ([32, 412])
        # self.prev_key = ktoken_pred
        # return rtoken_pred, ktoken_pred
