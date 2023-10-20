import os
import torch
import pretty_midi
import numpy as np
from pathlib import Path
from miditoolkit import MidiFile
from miditok import MIDILike, TokenizerConfig

# LupkerConfig = {
#     'notes': (128, 128),
#     'velocities': 1,
#     'time_shifts': 100,
#     'time_spacings_ms': 10
# }

# MIDILike is closest to This time with feeling (Oore et al.) 
config = TokenizerConfig(nb_velocities=1, use_chords=False, use_programs=True)
tokenizer = MIDILike(config)

maestro_dir = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/maestro-v3.0.0/'

tokenised = []
for root, dirs, files in os.walk(maestro_dir):
    for file in files:
        if file.endswith('mid') or file.endswith('midi'):
            midi = MidiFile(os.path.join(root, file))
            tokens = tokenizer(midi)
            # converted_back_midi = tokenizer(tokens) 
            print(file, len(tokens))
            tokenised.append(tokens)

np.save('datasets/maestro_midi.npy', tokenised, allow_pickle=True)