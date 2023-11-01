import torch
import torch.nn as nn
import torch.nn.functional as F

class StackMemory(nn.Module):
    def __init__(self, hidden_dim=400, stack_size=32):
        super(StackMemory, self).__init__()
        self.action_predictor = nn.Linear(hidden_dim, 3)
        self.D = nn.Parameter(torch.randn(1, hidden_dim))
        self.stack = None
        self.stack_depth = stack_size
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

    def forward(self, hidden_state):
        device = hidden_state.device
        batch_size, seq_len, _ = hidden_state.size()
        if self.stack is None:
            self.stack = torch.zeros(batch_size, self.stack_depth, self.hidden_dim).to(device)
        new_stacks = []
        for t in range(seq_len):
            # Extract the hidden_state at time t across all batches
            ht = hidden_state[:, t, :]
            # Predict the action to be taken on the stack
            action_prob = F.softmax(self.action_predictor(ht), dim=1)
            push_prob, pop_prob, no_op_prob = torch.chunk(action_prob, 3, dim=1)
            # Compute the value to be pushed onto the stack
            push_val = torch.sigmoid(torch.matmul(self.D.expand(batch_size, -1), ht.transpose(0, 1)).squeeze())
            # POP operation: Shift everything up, dropping the top element
            self.stack[:, :-1, :] = self.stack[:, 1:, :]
            # PUSH operation: Shift everything down
            self.stack[:, 1:, :] = self.stack[:, :-1, :]
            # Compute the new top of the stack based on the action probabilities
            self.stack[:, 0, :] = push_prob.squeeze() * push_val + pop_prob.squeeze() * self.stack[:, 1, :] + no_op_prob.squeeze() * self.stack[:, 0, :]
            new_stacks.append(self.stack)
        return torch.stack(new_stacks, dim=1)

class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.5) #we apply dropout of 0.5 on all the layers, excluding the recurrent connection

    def forward(self, x):
        rnn_out, (h, c) = self.rnn(x)
        token_pred = self.fc(rnn_out)
        token_pred = self.dropout(token_pred)
        return rnn_out, h, c, token_pred
    

class BachDuet(nn.Module):
    def __init__(self, token_dim=135, cpc=12, rhythm=8, hidden_dim=400, stack_size=32, key_dim=300, num_embedding_dim=50):
        super(BachDuet, self).__init__()
        self.embedding_midi = nn.Embedding(token_dim, num_embedding_dim)
        self.embedding_cpc = nn.Embedding(cpc, num_embedding_dim)
        self.embedding_rhythm = nn.Embedding(rhythm, num_embedding_dim)
        self.note_rnn = LSTMLayer(3 * num_embedding_dim, hidden_dim, num_layers=2)
        self.stack_memory = StackMemory(hidden_dim, stack_size)
        self.key_rnn = LSTMLayer(hidden_dim, key_dim, num_layers=1)
        self.prev_key = torch.zeros(1, cpc) # we initialize the previous key to 0

    def forward(self, xr, xcpc, xma):
        device = xr.device
        x_midi = self.embedding_midi(xma)
        x_cpc = self.embedding_cpc(xcpc)
        x_rhythm = self.embedding_rhythm(xr)
        print("embeddings", x_midi.shape, x_cpc.shape, x_rhythm.shape) #([32, 256, 50])([32, 256, 50])([32, 256, 50])
        x_embedded = torch.cat((x_midi, x_cpc, x_rhythm), dim=2)
        rnn_out, rh, rc, rtoken_pred = self.note_rnn(x_embedded)
        print("tp", rtoken_pred.shape)  #([32, 256, 150])
        print("rnn", rnn_out.shape)  #([32, 256, 400])
        context = self.stack_memory(rnn_out) # we use the last output of the RNN as the context
        print("c", context.shape) # ([32, 1, 400])
        #The input to the key LSTM, is a combination of the previous key, 
        # with the hidden state of the note LSTM unit.
        print("rnnh",rh.shape) # ([2, 32, 400])
        print("prev_key",self.prev_key.shape) # ([1, 12])
        combined_context = torch.cat((rh[-1], self.prev_key), dim=1).unsqueeze(1)
        print("cc",combined_context.shape) # ([32, 1, 412])
        krnn_out, kh, kc, ktoken_pred = self.key_rnn(combined_context)
        print("kp",ktoken_pred.shape)
        self.prev_key = ktoken_pred
        return token_pred, ktoken_pred


if __name__ == "__main__":
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset
    token_dim=135 + 13 + 10 # Midi_Artic + CPC + Rhythm and rests
    num_keys=24 # 24 keys in western music

    x_fake = torch.randint(0, token_dim, (32, 10, 3))
    y_fake_token = torch.randint(0, token_dim, (32, 10))  # Fake target tokens
    y_fake_key = torch.randint(0, num_keys, (32, 10))  # Fake target keys
    
    dataset = TensorDataset(x_fake, y_fake_token, y_fake_key)
    data_loader = DataLoader(dataset, batch_size=32)
    
    model = BachDuet()
    print(model)

    optimizer = Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    mistake_prob = 0.0

    for epoch in range(10):
        for x, y_token, y_key in data_loader:
            optimizer.zero_grad()
            token_pred, key_pred = model(x)
            loss = F.cross_entropy(token_pred.view(-1, token_dim), y_token.view(-1))
            loss += F.cross_entropy(key_pred.view(-1, num_keys), y_key.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")