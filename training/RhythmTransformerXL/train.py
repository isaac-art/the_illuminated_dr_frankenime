import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim

from miditok import MIDILike
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from x_transformers import XLAutoregressiveWrapper

from utils import p_
from utils.data import MIDIDataset
from utils.training import make_deterministic
from models.papers.nuttall_2021 import RhythmTransformerXL

p_()
make_deterministic()

device = 'mps'
batch_size = 8
num_steps = 10000
learning_rate = 1e-4
data_seq_len = 2048
midi_dataset = MIDIDataset(f'datasets/nuttall_groove_midi_{data_seq_len}.npy')
vocab_size = midi_dataset.vocab_size
tokenizer = MIDILike(params=Path(f"datasets/nuttall_groove_midi_{data_seq_len}.json"))

total_size = len(midi_dataset)
train_size = int(0.9 * total_size)  # 90% for training
val_size = total_size - train_size


train_dataset, val_dataset = random_split(midi_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=midi_collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=midi_collate_fn, drop_last=True)


print("Creating model...")
model = RhythmTransformerXL().to(device)
print(model)
p_()

print("Creating optimizer...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Creating loss function...")
criterion = torch.nn.CrossEntropyLoss() 


print("Training...")
current_step = 0
with tqdm(total=num_steps, desc="Training", position=0) as pbar:
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

            if current_step % 500 == 0:
                print(f"Step [{current_step}/{num_steps}] completed. Loss: {loss.item()}")
                torch.save(model.state_dict(), f'weights/rhythm_transformer_xl_{data_seq_len}.pth')

                # Switch to evaluation mode
                model.eval()
                
                # Initialize variables for validation metrics
                val_loss = 0
                val_steps = 0
                
                # Run validation
                with torch.no_grad():
                    for val_inputs, val_targets in val_loader:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_output = model(val_inputs)
                        val_loss_step = criterion(val_output.view(-1, vocab_size), val_targets.view(-1))
                        val_loss += val_loss_step.item()
                        val_steps += 1
                avg_val_loss = val_loss / val_steps
                print(f"Validation Loss after [{current_step}/{num_steps}] steps: {avg_val_loss}")
            
                model.train() #BACK TO TRAINING

        
# seg = torch.randint(0, 20000, (1, 4096)).cuda()  # sequence exceeding max length, automatically segmented and memory managed
# loss = xl_wrapper(seg)
# loss.backward()
# # then, after much training
# prime = seg[:, :1024]   # if prime exceeds max length, memory will be caught up before generating
# generated = xl_wrapper.generate(prime, 4096)  # (1, 4096)

