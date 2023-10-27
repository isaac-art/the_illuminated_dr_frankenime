import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import p_
from utils.data import NAESSEncoder
from utils.training import make_deterministic
from models.papers.faitas_2019 import ConvincingHarmony

p_()
make_deterministic()

ne = NAESSEncoder()
device = "mps"
num_chorales = 405
seq_len = 32
vocab_size = 130 # 0-127 MIDI notes + 128 for silence + 129 for repeat
n_epochs = 25000
batch_size = 128
lr = 1e-4

# LOAD DATA
# [seq_len, batch_size, vocab_size] [32, 128, 130]
inputs = np.load("datasets/ch_inputs.npy")
targets = np.load("datasets/ch_targets.npy")
inputs = torch.from_numpy(inputs).long()
targets = torch.from_numpy(targets).long()
print(inputs.shape, targets.shape) #torch.Size([4256, 33]) torch.Size([4256, 33])
# split 90/10 train/val
train_inputs = inputs[:int(num_chorales*0.9)]
train_targets = targets[:int(num_chorales*0.9)]
val_inputs = inputs[int(num_chorales*0.9):]
val_targets = targets[int(num_chorales*0.9):]

# group inputs, targets to load with DataLoader
train_loader = torch.utils.data.DataLoader(list(zip(train_inputs, train_targets)), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(list(zip(val_inputs, val_targets)), batch_size=batch_size, shuffle=True)


# LOAD MODEL
model = ConvincingHarmony().to(device)

print(model)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
tf = 0.5

# model.load_state_dict(torch.load("weights/ch_900.pt"))
# BEGIN TRAINING
for epoch in range(0, n_epochs):
    # TRAINING
    model.train()
    for batch_idx, data_batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_data = data_batch[0].to(device)
        target_data = data_batch[1].to(device)
        # input_data = data_batch[0][:, :-1].to(device)

        # target_data = data_batch[1][:, 1:].to(device)
        # target_data = data_batch[0][:, 1:].to(device)

        output = model(input_data, target_data, teacher_forcing_ratio=tf)
        output = output.permute(0, 2, 1)
        loss = criterion(output, target_data)
        # print("Loss", loss)
        loss.backward()

        # #monitor grad norms
        # total_norm = 0
        # for p in model.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # print("Total norm", total_norm)

        # # # clip grads
        clip_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        print("Epoch: {} Batch: {} Loss: {}".format(epoch, batch_idx, loss.item()), end='\r')
    print("Epoch: {} Loss: {}".format(epoch, loss.item()))
    train_loss = loss.item()
    # tf = tf * 0.99 # gradually reduce teacher forcing ratio
    if epoch % 100 == 0:
        # save
        torch.save(model.state_dict(), f"weights/ch_{epoch}.pt")
        # VALIDATION
        with torch.no_grad():
            model.eval()
            random_batch_id = np.random.randint(0, len(val_loader))
            data_batch = list(val_loader)[random_batch_id]

            input_data = data_batch[0].to(device)
            target_data = data_batch[1].to(device)
            # input_data = data_batch[0][:, :-1].to(device)

            # target_data = data_batch[1][:, 1:].to(device)
            # target_data = data_batch[0][:, 1:].to(device)
            
            print(input_data[0])
            print(target_data[0])

            output = model(input_data)
            output = output.permute(0, 2, 1)
            loss = criterion(output, target_data)
            print("Validation Loss: {}".format(loss.item()), end="\r")
            sample = output[0].argmax(dim=0).cpu().numpy()
            print(sample)
            midi_sample = ne.decode(sample)
            midi_sample.dump(f"samples/ch/{epoch}.mid")
        print("Validation Loss: {}".format(loss.item()))
        print("-"*120)
    vali_loss = loss.item()
    if vali_loss > train_loss * 5:
        print("Validation loss is too high, exiting")
        break
torch.save(model.state_dict(), f"weights/ch_{epoch}.pt")