import os
import torch
import pretty_midi
import numpy as np
from pathlib import Path
from miditoolkit import MidiFile
from miditok import MIDILike, TokenizerConfig

from utils.data import WarrenMIDIMarkov

data_root = "/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove"
warrenMIDIMarkov = WarrenMIDIMarkov(quantize_to='16th')
for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith(".mid"):
            res, encoded = warrenMIDIMarkov.encode(os.path.join(root, file))
            if res: 
                print(encoded)