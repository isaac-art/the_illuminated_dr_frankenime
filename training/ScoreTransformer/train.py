import os
import torch
import numpy as np
import torch.optim as optim
from miditoolkit import MidiFile
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from miditok import MIDILike, TokenizerConfig

from models.papers import ScoreTransformer
from utils.data import MIDIDataset, midi_collate_fn

if __name__ == "__main__":
    device = "mps"
    model_params = {
        "d_model": 512, "max_seq_len": 256
    }
    train_params = {
        "batch_size": 32, "num_epochs": 100, "num_steps": 1000,
        "lr": 1e-4, "val_interval": 500, "save_interval": 1000
    }

    data_dir = "datasets/maestro-v2.0.0"
    token_np = f"datasets/lupker_maestro_midi_{model_params['max_seq_len']}.npy"
    midi_dataset = MIDIDataset(token_np)
    vocab_size = midi_dataset.vocab_size
    total_size = len(midi_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(midi_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, collate_fn=midi_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"], shuffle=False, collate_fn=midi_collate_fn, drop_last=True)

    model = ScoreTransformer(num_tokens=vocab_size, max_seq_len=model_params["max_seq_len"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_params["lr"])
    criterion = torch.nn.CrossEntropyLoss() 

    for epoch in range(train_params["num_epochs"]):
        model.train()
        for step, (inputs, targets) in enumerate(train_loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{train_params['num_epochs']}, Step {step+1}, Loss {loss.item()}", end="\r") 
            
            if step % train_params["val_interval"]+1 == 0:
                model.eval()
                val_loss = 0
                val_steps = 0
                with torch.no_grad():
                    for val_inputs, val_targets in val_loader:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_output = model(val_inputs)
                        val_loss_step = criterion(val_output.view(-1, vocab_size), val_targets.view(-1))
                        val_loss += val_loss_step.item()
                        val_steps += 1
                        avg_val_loss = val_loss / val_steps
                        print("Loss/val", avg_val_loss, val_steps)
                model.train() #reset

            if step % train_params["save_interval"] == 0:
                torch.save(model.state_dict(), f"archive/score_transformer_{model_params['max_seq_len']}_1.pt")
                print("saved")
        torch.save(model.state_dict(), f"archive/score_transformer_{model_params['max_seq_len']}.pt")
        print("saved")
            
    torch.save(model.state_dict(), f"archive/score_transformer_{model_params['max_seq_len']}.pt")
    print("saved")
