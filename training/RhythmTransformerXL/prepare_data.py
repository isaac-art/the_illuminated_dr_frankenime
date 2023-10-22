import os
import torch
import pretty_midi
import numpy as np
from pathlib import Path
from miditoolkit import MidiFile
from miditok import MIDILike, TokenizerConfig
from miditok.utils import convert_ids_tensors_to_list

# 4 velocities, 9 pitches, 
NuttallConfig = {
    'nb_velocities': 4,
    'nb_tempos': 1,
    'use_chords': False,
    'use_programs': False,
    'use_tempos': False,
    'use_time_signatures': False,
    'one_token_stream': True,
}

# MIDILike is closest to This time with feeling (Oore et al.) 
config = TokenizerConfig(**NuttallConfig)
tokenizer = MIDILike(config)

maestro_dir = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'

seq_len = 2048
tokenised_stream = []
file_count = 0
for root, dirs, files in os.walk(maestro_dir):
    for file in files:
        if file.endswith('mid') or file.endswith('midi'):
            file_count += 1
            print(file_count, end="\r")
            midi = MidiFile(os.path.join(root, file))
            tokens = tokenizer(midi)
            # chunk into multiple sequences of len seq_len (discard remainder)
            for i in range(0, len(tokens[0].ids), seq_len):
                s = tokens[0].ids[i:i+seq_len]
                if len(s) == seq_len:
                    # print(len(s))
                    tokenised_stream += DIVISION TOKEN HERE
                    tokenised_stream += tokens[0].ids[i:i+seq_len]
            # tokenised.append(tokens[0].ids)

# shoudl be one long stream

tokenizer.save_params(f'datasets/nuttall_groove_midi_{seq_len}.json')
np.save(f'datasets/nuttall_groove_midi_{seq_len}.npy', tokenised, allow_pickle=True)

#max([max(seq) for seq in self.tokenized_sequences]) + 1
print("vocab size:", max([max(seq) for seq in tokenised]) + 1)