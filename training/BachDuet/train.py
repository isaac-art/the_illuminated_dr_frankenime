import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from models.papers.benatatos_2020 import BachDuet
from utils.training import make_deterministic
from utils.data import BachDuetData

make_deterministic()

bdd = BachDuetData()
bdd_inputs = np.load("datasets/bdd_inputs.npy", allow_pickle=True)
bdd_targets = np.load("datasets/bdd_targets.npy", allow_pickle=True)
# print(bdd_inputs.shape, bdd_targets.shape) #(4023, 256, 3) (4023, 256, 3)

train_split = int(0.8 * bdd_inputs.shape[0])
bdd_train_inputs = bdd_inputs[:train_split]
bdd_train_targets = bdd_targets[:train_split]
bdd_test_inputs = bdd_inputs[train_split:]
bdd_test_targets = bdd_targets[train_split:]

# print(bdd_train_inputs.shape, bdd_train_targets.shape) #(3218, 256, 3) (3218, 256, 3)

train_dataset = TensorDataset(torch.from_numpy(bdd_train_inputs), torch.from_numpy(bdd_train_targets))
test_dataset = TensorDataset(torch.from_numpy(bdd_test_inputs), torch.from_numpy(bdd_test_targets))
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

device = "mps"
n_epochs = 10
lr = 1e-3
rhythm_classes = 8
cpc_classes = 12
midi_artic_classes = 135
model = BachDuet()
opt = Adam(model.parameters(), lr=lr)
crit = torch.nn.CrossEntropyLoss()

print(model)
model.to(device)

# Rhythm, CPC, Midi_Artic
for i in tqdm(range(n_epochs)):
    for batch in train_loader:
        x, y = batch
        xr = x[:, :, 0].long().to(device)
        xcpc = x[:, :, 1].long().to(device)
        xma = x[:, :, 2].long().to(device)
        y = y.long().to(device)
        opt.zero_grad()
        print("input", xr.shape, xcpc.shape, xma.shape)
        token_pred, key_pred = model(xr, xcpc, xma)
        token_loss = crit(token_pred.reshape(-1, token_pred.shape[-1]), y.reshape(-1))
        key_loss = crit(key_pred.reshape(-1, key_pred.shape[-1]), y.reshape(-1))
        loss = token_loss + key_loss
        loss.backward()
        opt.step()
        print("Epoch", i, "Loss", loss.item(), "Token Loss", token_loss.item(), "Key Loss", key_loss.item(), end="\r")
    print("Epoch", i, "Loss", loss.item())
