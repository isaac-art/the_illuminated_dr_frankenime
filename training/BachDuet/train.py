import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from models.papers import BachDuet
from utils.training import make_deterministic
from utils.data import BachDuetData

make_deterministic()

bdd = BachDuetData()
bdd_inputs = np.load("datasets/bdd_inputs_32.npy", allow_pickle=True) #32 len seqs, bdd_inputs ==256 len
bdd_targets = np.load("datasets/bdd_targets_32.npy", allow_pickle=True)
# print(bdd_inputs.shape, bdd_targets.shape) #(4023, 256, 3) (4023, 256, 3)

train_split = int(0.8 * bdd_inputs.shape[0])
bdd_train_inputs = bdd_inputs[:train_split]
bdd_train_targets = bdd_targets[:train_split]
bdd_test_inputs = bdd_inputs[train_split:]
bdd_test_targets = bdd_targets[train_split:]

# print(bdd_train_inputs.shape, bdd_train_targets.shape) #(3218, 256, 3) (3218, 256, 3)

batch_size=256
train_dataset = TensorDataset(torch.from_numpy(bdd_train_inputs), torch.from_numpy(bdd_train_targets))
test_dataset = TensorDataset(torch.from_numpy(bdd_test_inputs), torch.from_numpy(bdd_test_targets))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = "mps"
n_epochs = 10000
lr = 1e-4
rhythm_classes = 8
cpc_classes = 24 # 12 min 12 maj keys
midi_artic_classes = 135
model = BachDuet()
opt = Adam(model.parameters(), lr=lr)
crit = torch.nn.CrossEntropyLoss()

print(model)
model.to(device)

for i in tqdm(range(n_epochs)):
    model.train()        
    for batch in train_loader:
        x, y = batch
        # input soprano
        xr = x[:, :, 0].long().to(device) #rhythm
        xcpc = x[:, :, 1].long().to(device) #cpc key
        xma = x[:, :, 2].long().to(device) #midi sequence
        # target bass
        yr = y[:, :, 0].long().to(device) #rhythm
        ycpc = y[:, :, 1].long().to(device) #cpc key
        yma = y[:, :, 2].long().to(device) #midi sequence
        opt.zero_grad()
        token_pred, token_pred_sm, key_pred, key_pred_sm = model(xr, xcpc, xma)
        note_loss = crit(token_pred.reshape(-1, token_pred.shape[-1]), yma.reshape(-1))
        key_loss = crit(key_pred.reshape(-1, key_pred.shape[-1]), ycpc.reshape(-1))
        loss = note_loss + key_loss
        loss.backward()
        opt.step()
        print("Epoch", i, "Loss", loss.item(), "Note Loss", note_loss.item(),  "Key Loss", key_loss.item(), end="\r") #
    print("Epoch", i, "Loss", loss.item())
    train_loss = loss.item()
    if i % 50 == 0 and i != 0:
        # save model
        torch.save(model.state_dict(), f"archive/bachduet_32_{i}.pt")
        # test model
        with torch.no_grad():
            model.eval()
            for batch in test_loader:
                x, y = batch
                xr = x[:, :, 0].long().to(device)
                xcpc = x[:, :, 1].long().to(device)
                xma = x[:, :, 2].long().to(device)
                yr = y[:, :, 0].long().to(device)
                ycpc = y[:, :, 1].long().to(device)
                yma = y[:, :, 2].long().to(device)
                token_pred, token_pred_sm, key_pred, key_pred_sm = model(xr, xcpc, xma)
                note_loss = crit(token_pred.reshape(-1, token_pred.shape[-1]), yma.reshape(-1))
                key_loss = crit(key_pred.reshape(-1, key_pred.shape[-1]), ycpc.reshape(-1))
                loss = note_loss + key_loss
                print("Test Loss", loss.item(), end="\r")
            print("Test Loss", loss.item())
            val_loss = loss.item()
            # if val loss is more than 10% greater than train loss, reduce lr
            if val_loss > 1.1 * train_loss:
                lr = lr / 10
                opt = Adam(model.parameters(), lr=lr)
                print("Reducing lr to", lr)
    print("-"*120)
torch.save(model.state_dict(), f"archive/bachduet_32_{i}.pt")