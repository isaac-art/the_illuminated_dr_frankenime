import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.papers.lupker_2021 import ScoreTransformer

from utils import p_
from utils.training import make_deterministic

p_()
make_deterministic()

device = "cpu"

d_model = 512
nhead = 8
num_layers = 8
dim_feedforward = 2048
max_len = 10000  # large number for max seq len
batch_size = 16
num_steps = 250000
learning_rate = 0.001
seq_len = 2048

print("Loading data...")
tokenized_sequences = np.load('datasets/maestro_midi.npy', allow_pickle=True) # typ np.object_.
num_midi_tracks = tokenized_sequences.shape[0]
print("Num tokenised MIDI Tracks: ", num_midi_tracks)

vocab_size = np.max([np.max([np.max(track) for track in tokenized_sequences]) for track in tokenized_sequences]) + 1
print("Vocab size: ", vocab_size)

def one_hot_encode(sequence, vocab_size):
    return F.one_hot(torch.tensor(sequence), num_classes=vocab_size)

one_hot_sequences = [one_hot_encode(seq, vocab_size) for seq in tokenized_sequences]

def create_inputs_targets(one_hot_sequences):
    inputs = [seq[:-1] for seq in one_hot_sequences]
    targets = [seq[1:] for seq in one_hot_sequences]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)    
    return padded_inputs, padded_targets

inputs, targets = create_inputs_targets(one_hot_sequences)
print("Inputs shape: ", inputs.shape)
print("Targets shape: ", targets.shape)

exit()

print("Creating model...")
model = ScoreTransformer(d_model, nhead, num_layers, dim_feedforward, vocab_size, max_len)
print(model)
p_()

print("Creating optimizer...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Creating loss function...")
criterion = torch.nn.CrossEntropyLoss() 

print("Creating data loader...")
train_loader = DataLoader(
    OneHotDataset(loaded_encoded_midi, vocab_size),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: auto_reg_collate_fn(batch, vocab_size)
)

print("Training...")
current_step = 0
# PAPER: The model plateaued at a loss of ~1.7 during training after approximately 250,000 steps.
memory = torch.zeros(batch_size, seq_len, d_model).to(device)  # Adjust dimensions as needed

# Training loop
while current_step < num_steps:
    for batch_idx, (input_data, target_data) in enumerate(train_loader):
        if current_step >= num_steps:
            break

        # Move data to device
        print(input_data.shape, target_data.shape)
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        # mask all pad tokens (vocab_size) 
        # mask = (input_data.sum(dim=2) != vocab_size).unsqueeze(1).unsqueeze(2).to(device)
        output = model(input_data, memory)
        
        # Compute loss and gradients
        loss = criterion(output.view(-1, vocab_size), target_data.view(-1))
        loss.backward()

        # Update parameters
        optimizer.step()

        # Step counter
        current_step += 1

        # Logging
        if current_step % 1000 == 0:
            print(f"Step [{current_step}/{num_steps}] completed. Loss: {loss.item()}")

# save 
torch.save(model.state_dict(), 'weights/score_transformer.pth')


# tensors = {
#     "embedding": torch.zeros((2, 2)),
#     "attention": torch.zeros((2, 3))
# }
# save_file(tensors, "model.safetensors")