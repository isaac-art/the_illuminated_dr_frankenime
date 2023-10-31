import torch
import torch.nn as nn
import torch.nn.functional as F

class MainRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MainRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.5) #we apply dropout of 0.5 on all the layers, excluding the recurrent connection

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        token_pred = self.fc(rnn_out)
        token_pred = self.dropout(token_pred)
        return token_pred

class StackMemory(nn.Module):
    def __init__(self, hidden_dim, stack_size):
        super(StackMemory, self).__init__()
        self.stack = torch.zeros(stack_size, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, rnn_out):
        attention_score = self.attention(rnn_out)
        attention_weight = F.softmax(attention_score, dim=1)
        context = torch.sum(attention_weight * self.stack, dim=1)
        context = self.dropout(context)
        return context.unsqueeze(1)
    
    def push(self, rnn_out):
        self.stack = torch.cat((self.stack[1:], rnn_out), dim=0)

    def pop(self):
        return self.stack[-1]
    
   
class KeyRNN(nn.Module):
    def __init__(self, hidden_dim, key_dim, num_keys):
        super(KeyRNN, self).__init__()
        self.key_rnn = nn.LSTM(input_size=hidden_dim, hidden_size=key_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(key_dim, num_keys)
        self.dropout = nn.Dropout(0.5)

    def forward(self, rnn_out):
        key_rnn_out, _ = self.key_rnn(rnn_out)
        key_pred = self.fc(key_rnn_out)
        key_pred = self.dropout(key_pred)
        return key_pred
    
class BachDuet(nn.Module):
    def __init__(self, token_dim=135, hidden_dim=400, stack_size=32, key_dim=300, num_keys=24, num_embedding_dim=50):
        super(BachDuet, self).__init__()
        self.embedding_midi = nn.Embedding(token_dim, num_embedding_dim)
        self.embedding_cpc = nn.Embedding(token_dim, num_embedding_dim)
        self.embedding_rhythm = nn.Embedding(token_dim, num_embedding_dim)
        self.main_rnn = MainRNN(3 * num_embedding_dim, hidden_dim)
        self.stack_memory = StackMemory(hidden_dim, stack_size)
        self.key_rnn = KeyRNN(hidden_dim, key_dim, num_keys)

    def forward(self, x):
        x_midi = self.embedding_midi(x[:,:,0])
        x_cpc = self.embedding_cpc(x[:,:,1])
        x_rhythm = self.embedding_rhythm(x[:,:,2])
        print("embeddings", x_midi.shape, x_cpc.shape, x_rhythm.shape)
        x_embedded = torch.cat((x_midi, x_cpc, x_rhythm), dim=2)
        print("cat", x_embedded.shape)
        token_pred = self.main_rnn(x_embedded)
        print("tp", token_pred.shape)
        context = self.stack_memory(token_pred)
        print("c",context.shape)
        combined_context = token_pred + context
        print("cc",combined_context.shape)
        key_pred = self.key_rnn(combined_context)
        print("kp",key_pred.shape)
        return token_pred, key_pred


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