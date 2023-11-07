import torch
from tqdm import tqdm
import torch.optim as optim
from safetensors.torch import save_file
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.papers.lupker_2021 import ScoreTransformer
from utils import p_
from utils.data import MIDIDataset, midi_collate_fn
from utils.training import make_deterministic

p_()
make_deterministic()


writer = SummaryWriter('training/_runs/score_transformer')
device = "mps"
d_model = 512
batch_size = 8
num_steps = 10000 #250000 steps in paper
learning_rate = 1e-4
max_seq_len = 256 
chunk = True
if chunk: midi_dataset = MIDIDataset(f'datasets/lupker_maestro_midi_{max_seq_len}.npy')
else: 
    max_seq_len = 2048 #??
    midi_dataset = MIDIDataset(f'datasets/lupker_maestro_midi_full.npy')

vocab_size = midi_dataset.vocab_size
print("Vocab size:", vocab_size) #245

total_size = len(midi_dataset)
train_size = int(0.9 * total_size)  # 90% for training
val_size = total_size - train_size

train_dataset, val_dataset = random_split(midi_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=midi_collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=midi_collate_fn, drop_last=True)


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

            writer.add_scalar('Loss/train', loss.item(), current_step)
            if current_step % 1000 == 0:
                print(f"Step [{current_step}/{num_steps}] completed. Loss: {loss.item()}")
                torch.save(model.state_dict(), f'weights/score_transformer_{max_seq_len}.pth')
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
                
                writer.add_scalar('Loss/val', avg_val_loss, current_step)
                model.train() #BACK TO TRAINING

# tensors = {
#     "embedding": torch.zeros((2, 2)),
#     "attention": torch.zeros((2, 3))
# }
# save_file(tensors, "model.safetensors")