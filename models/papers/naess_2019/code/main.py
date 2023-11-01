import torch.nn as nn

from models.core import BaseModel

class PhysicallyIntelligentRNN(BaseModel):
    def __init__(self, vocab_size=130, embedding_dim=256, hidden_dim=256, num_layers=2, dropout=0.1):
        super(PhysicallyIntelligentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        # x = F.softmax(x, dim=2) # cant train with this, pytorch crossentropy already applies so maybe why
        return x, hidden
