import torch
import numpy as np
import torch.optim as optim

from models.papers import PhysicallyIntelligentRNN

device = "mps"
batch_size = 128
model = PhysicallyIntelligentRNN().to(device)
# # load state from prev train
print(model)

data = np.load("datasets/pii.npy")
data = torch.from_numpy(data).int()

train_data = data[:int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(len(train_loader), len(test_loader))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

epochs = 1000
start_epoch = 0
if start_epoch > 0:
    model.load_state_dict(torch.load(f"archive/pii_{start_epoch}.pt", map_location=device))
for epoch in range(start_epoch, epochs):
    for batch_idx, data_batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_data = data_batch[:, :-1].to(device)
        target_data = data_batch[:, 1:].to(device)
        output, _ = model(input_data)
        output = output.permute(0, 2, 1)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        print("Epoch: {} Batch: {} Loss: {}".format(epoch, batch_idx, loss.item()), end="\r")
    print("Epoch: {} Loss: {}".format(epoch,    loss.item()))
    if epoch % 50 == 0:
        torch.save(model.state_dict(), f"archive/pii_{epoch}.pt")
        # validation
        with torch.no_grad():
            model.eval()
            for batch_idx, data_batch in enumerate(test_loader):
                input_data = data_batch[:, :-1].to(device)
                target_data = data_batch[:, 1:].to(device)
                output, _ = model(input_data)
                output = output.permute(0, 2, 1)
                loss = criterion(output, target_data)
                print("Validation Loss: {}".format(loss.item()), end="\r")
            model.train()
        print("Validation Loss: {}".format(loss.item()))
        # if validation loss is too far from training loss, then we are overfitting
        if loss.item() > 1.5:
            break 
torch.save(model.state_dict(), f"archive/pii_{epoch}.pt")