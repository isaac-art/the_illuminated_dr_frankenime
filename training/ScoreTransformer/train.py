import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
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
batch_size = 8
num_steps = 250000
learning_rate = 0.001
seq_len = 2048 -1 # -1 as we are predicting next token
midi_dataset = MIDIDataset('datasets/lupker_maestro_midi.npy')
vocab_size = midi_dataset.vocab_size


print("Vocab size:", vocab_size)

total_size = len(midi_dataset)
train_size = int(0.8 * total_size)  # 80% for training
val_size = total_size - train_size

train_dataset, val_dataset = random_split(midi_dataset, [train_size, val_size])

def collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


print("Creating model...")
model = ScoreTransformer(vocab_size=vocab_size, max_len=seq_len, dim_feedforward=seq_len).to(device)
print(model)
p_()

print("Creating optimizer...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Creating loss function...")
criterion = torch.nn.CrossEntropyLoss() 

print("Training...")
current_step = 0
num_steps = 2000

# PAPER: The model plateaued at a loss of ~1.7 during training after approximately 250,000 steps.
memory = torch.zeros(seq_len, batch_size, d_model).to(device)  # Just Zeros as we are only training decoder, but need to pass something to torch transofmrerdecoderlayer 


with tqdm(total=num_steps, desc="Training", position=0) as pbar:
    current_step = 0
    while current_step < num_steps:
        for inputs, targets in train_loader:
            if current_step >= num_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            tgt_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0)).to(device)

            optimizer.zero_grad()
            output = model(inputs, memory, tgt_mask=tgt_mask)

            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            current_step += 1
            pbar.update(1)  # Update tqdm progress by one step

            if current_step % 100 == 0:
                print(f"Step [{current_step}/{num_steps}] completed. Loss: {loss.item()}")

    torch.save(model.state_dict(), 'weights/score_transformer.pth')

# tensors = {
#     "embedding": torch.zeros((2, 2)),
#     "attention": torch.zeros((2, 3))
# }
# save_file(tensors, "model.safetensors")