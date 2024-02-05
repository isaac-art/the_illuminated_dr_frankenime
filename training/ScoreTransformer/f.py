import os
import torch
import subprocess
import numpy as np
import torch.optim as optim
from miditoolkit import MidiFile
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from miditok import MIDILike, TokenizerConfig
from torch.utils.tensorboard import SummaryWriter


from training.base_trainer import BaseTrainer
from models.papers import ScoreTransformer
from utils.data import MIDIDataset, midi_collate_fn

class ScoreTransformerTrainer(BaseTrainer):
    def __init__(self, config, *args, **kwargs) -> None:
        super(ScoreTransformerTrainer, self).__init__(*args, **kwargs)
        self.model_format = config.model_format
        self.device = config.device
        self.job_id = config.job_id
        self.writer = SummaryWriter(f'archive/summary/{config.job_id}')        
        self.model_params = {
            "d_model": 512, "max_seq_len": 256
        }
        self.train_params = {
            "batch_size": 8, "num_epochs": 100, "num_steps": 100,
            "lr": 1e-4, "val_interval": 500, "save_interval": 1000
        }
        self.data_dir = config.data_dir
        self.token_np = config.token_np if config.token_np else self.prepare_data()
        self.midi_dataset = MIDIDataset(self.token_np)
        self.vocab_size = self.midi_dataset.vocab_size
        total_size = len(self.midi_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = random_split(self.midi_dataset, [train_size, val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_params["batch_size"], shuffle=True, collate_fn=midi_collate_fn, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.train_params["batch_size"], shuffle=False, collate_fn=midi_collate_fn, drop_last=True)

        self.model = ScoreTransformer(num_tokens=self.vocab_size, max_seq_len=self.model_params["max_seq_len"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_params["lr"])
        self.criterion = torch.nn.CrossEntropyLoss() 

        config_dict = {
            "model": "ScoreTransformer",
            "model_format": self.model_format,
            "device": self.device,
            "job_id": self.job_id,
            "writer": f'archive/summary/{self.job_id}',
            "model_params": self.model_params,
            "train_params": self.train_params,
            "data_dir": self.data_dir,
            "token_np": self.token_np,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
        }
        self.save_config(**config_dict)
        print(f"JOB ID: {self.job_id}")
        

    def prepare_data(self):
        LupkerConfig = {
            'nb_velocities': 1,
            'nb_tempos': 1,
            'use_chords': False,
            'use_programs': False,
            'use_tempos': False,
            'use_time_signatures': False,
            'one_token_stream': True,
        }
        config = TokenizerConfig(**LupkerConfig)
        tokenizer = MIDILike(config)
        tokenised = []
        file_count = 0
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('mid') or file.endswith('midi'):
                    file_count += 1
                    print(file_count, end="\r")
                    midi = MidiFile(os.path.join(root, file))
                    tokens = tokenizer(midi)
                    for i in range(0, len(tokens[0].ids), self.model_params["max_seq_len"]):
                        s = tokens[0].ids[i:i+self.model_params["max_seq_len"]]
                        if len(s) == self.model_params["max_seq_len"]:
                            tokenised.append(tokens[0].ids[i:i+self.model_params["max_seq_len"]])
        p = f'datasets/lupker_maestro_midi_{self.job_id}'
        tokenizer.save_params(f'{p}.json')
        np.save(f'{p}.npy', tokenised, allow_pickle=True)
        return f'{p}.npy'

    def train_model(self,  *args, **kwargs):
        self.writer.flush()
        tensorboard_process = subprocess.Popen(["tensorboard", f"--logdir=archive/summary/{self.job_id}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Started TensorBoard with PID {tensorboard_process.pid}")
        try:
            for epoch in range(self.train_params["num_epochs"]):
                self.model.train()
                for step, (inputs, targets) in enumerate(self.train_loader):
                    print(f"Epoch {epoch+1}/{self.train_params['num_epochs']}, Step {step+1}", end="\r")
                    self.train_step(inputs, targets, (step+1)*(epoch+1))
                    if step % self.train_params["val_interval"]+1 == 0: self.val((step+1)*(epoch+1))
                    if step % self.train_params["save_interval"]+1 == 0: self.save_model()
        finally:
            tensorboard_process.terminate()
            print(f"Stopped TensorBoard with PID {tensorboard_process.pid}")
    
    def train_step(self, inputs, targets, step):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(inputs)
        self.loss = self.criterion(output.view(-1, self.vocab_size), targets.view(-1))
        self.loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('Loss/train', self.loss.item(), step)
        return 
    
    def val(self):
        print("Validating...")
        self.model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for val_inputs, val_targets in self.val_loader:
                val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)
                val_output = self.model(val_inputs)
                val_loss_step = self.criterion(val_output.view(-1, self.vocab_size), val_targets.view(-1))
                val_loss += val_loss_step.item()
                val_steps += 1
                avg_val_loss = val_loss / val_steps
                self.writer.add_scalar('Loss/val', avg_val_loss, val_steps)
        return

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=str, default='test')
    parser.add_argument('--data_dir', type=str, default='datasets/maestro-v2.0.0')
    parser.add_argument('--token_np', type=str, default='datasets/lupker_maestro_midi_128.npy')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--model_format', type=str, default='pth')
    args = parser.parse_args()
    trainer = ScoreTransformerTrainer(args)
    trainer.train_model()