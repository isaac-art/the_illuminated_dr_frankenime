import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from models.papers.lupker_2021 import ScoreTransformer

from utils import p_
from utils.training import make_deterministic

p_()
make_deterministic()

class MIDIDataset(Dataset):
    def __init__(self, np_file_path):
        self.tokenized_sequences = np.load(np_file_path, allow_pickle=True)
        self.vocab_size = self.get_vocab_size()

    def get_vocab_size(self):
        return max([max(seq) for seq in self.tokenized_sequences]) + 1

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, idx):
        sequence = self.tokenized_sequences[idx]
        inputs = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        return inputs, targets


device = "mps"
d_model = 512
batch_size = 16
num_steps = 250000
learning_rate = 1e-4
max_seq_len = 256 
midi_dataset = MIDIDataset(f'datasets/lupker_maestro_midi_{max_seq_len}.npy')
vocab_size = midi_dataset.vocab_size
print("Vocab size:", vocab_size) #245

total_size = len(midi_dataset)
train_size = int(0.8 * total_size)  # 80% for training
val_size = total_size - train_size

train_dataset, val_dataset = random_split(midi_dataset, [train_size, val_size])

def collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)


print("Creating model...")
model = ScoreTransformer(num_tokens=vocab_size, max_seq_len=max_seq_len).to(device)
print(model)
p_()

print("Creating optimizer...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Creating loss function...")
criterion = torch.nn.CrossEntropyLoss() 

print("Training...")
current_step = 0
# PAPER: The model plateaued at a loss of ~1.7 during training after approximately 250,000 steps.
with tqdm(total=num_steps, desc="Training", position=0) as pbar:
    current_step = 0
    while current_step < num_steps:
        for inputs, targets in train_loader:
            if current_step >= num_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            current_step += 1
            pbar.update(1)  # Update tqdm progress by one step

            if current_step % 1000 == 0:
                print(f"Step [{current_step}/{num_steps}] completed. Loss: {loss.item()}")
                torch.save(model.state_dict(), f'weights/score_transformer_{max_seq_len}.pth')

# tensors = {
#     "embedding": torch.zeros((2, 2)),
#     "attention": torch.zeros((2, 3))
# }
# save_file(tensors, "model.safetensors")