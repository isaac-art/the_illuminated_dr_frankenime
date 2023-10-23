import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from x_transformers import XLAutoregressiveWrapper

from utils import p_
from utils.data import NuttallGrooveTokenizer
from utils.training import make_deterministic
from models.papers.nuttall_2021 import RhythmTransformerXL

p_()
make_deterministic()

ngt = NuttallGrooveTokenizer()
device = 'mps'
num_steps = 30000
max_seq_len = 128
max_mem_len = int(max_seq_len * 1.5)
learning_rate = 1e-4

data = f'datasets/nuttall_groove_encoded_test_train_val.npy' #(contains three seqs test, train, val)
test, train, val = np.load(data, allow_pickle=True)
# print(len(test), len(train), len(val))
# test: 147491, train: 1198098, val: 150980 - one long stream of tokens for each
# print(set(train)) # {0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41}

vocab_size = len(set(train)) # 40

print("Creating model...")
model = RhythmTransformerXL(num_tokens=vocab_size, max_seq_len=max_seq_len, max_mem_len=max_mem_len).to(device)
print(model)
p_()

print("Creating optimizer...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training...")
current_step = 0

xl_wrapper = XLAutoregressiveWrapper(model)

# train_torch = torch.tensor(train).unsqueeze(0).to(device)
# test_torch = torch.tensor(test[:1000]).unsqueeze(0).to(device)
# val_torch = torch.tensor(val[:1000]).unsqueeze(0).to(device)

chunk_size = max_seq_len * 2
train_chunks = [train[i:i + chunk_size] for i in range(0, len(train), chunk_size)]
# discard last chunk if it's not the right size
if len(train_chunks[-1]) != chunk_size:
    train_chunks = train_chunks[:-1]
train_dataset = TensorDataset(torch.tensor(train_chunks))
train_loader = DataLoader(train_dataset, batch_size=32)

val_lists = [[]]
j = 0
for i in range(len(val)):
    if val[i] == 0:
        # start new list
        j+=1
        val_lists.append([])
    else:
        val_lists[j].append(val[i])

# model.train()
# for step in tqdm(range(num_steps)):
#     optimizer.zero_grad()

#     loss = xl_wrapper(train_torch)
#     loss.backward()
#     optimizer.step()
accumulation_steps = 4  
model.train()

pbar = tqdm(range(num_steps))
for step in pbar:
    optimizer.zero_grad()
    batch = next(iter(train_loader))
    batch = batch[0].to(device)  # Assuming batch is a tuple containing one tensor
    loss = xl_wrapper(batch)
    loss = loss / accumulation_steps
    loss.backward()
    pbar.set_description(f"Loss: {loss.item()}")

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    if step % 1000 == 0 and step > 0:
        print(f"Step {step} | Loss: {loss.item()}")
        # save
        torch.save(model.state_dict(), f"weights/rhythm_transformer_{max_seq_len}_{max_mem_len}.pt")

        model.eval()
        # validate on a random 20 int chunk of random val list
        val_list = val_lists[np.random.randint(0, len(val_lists))]
        val_chunk_start = np.random.randint(0, len(val_list)-20)
        val_chunk = val_list[val_chunk_start:val_chunk_start+20]
        val_chunk_torch = torch.tensor(val_chunk).unsqueeze(0).to(device)
        gen = xl_wrapper.generate(start_tokens=val_chunk_torch, seq_len=max_seq_len*2, eos_token=0, temperature=1.0)
        print(gen)
        midif = ngt.decode(gen[0].cpu().numpy().tolist())
        midif.write(f'samples/rt/test_{step}.mid')
        model.train()

torch.save(model.state_dict(), f"weights/rhythm_transformer_{max_seq_len}_{max_mem_len}.pt")