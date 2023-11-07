import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class MIDIDataset(Dataset):
    def __init__(self, np_file_path):
        self.tokenized_sequences = np.load(np_file_path, allow_pickle=True)
        self.vocab_size = self.get_vocab_size()

    def get_vocab_size(self):
        return max([max(seq) for seq in self.tokenized_sequences]) + 1

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, idx):
        sequence = self.tokenized_sequences[idx]
        inputs = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        return inputs, targets


def midi_collate_fn(batch, pad=True, pad_value=0):
    inputs, targets = zip(*batch)
    if pad:
        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_value)
        targets = pad_sequence(targets, batch_first=True, padding_value=pad_value)
    return inputs, targets