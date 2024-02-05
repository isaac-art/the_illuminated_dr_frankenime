import os
import csv
import torch
import pretty_midi
import numpy as np
from pathlib import Path
from collections import deque

from utils.data import NuttallGrooveTokenizer 

# train 378, test 77, validation 48 split
# approx~ 75% train, 15% test, 10% validation
ngt = NuttallGrooveTokenizer()

test, train, val = [], [], []
root = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'
groove_data =  f'{root}/info.csv'
with open(groove_data, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)
    for row in data:
        split = row['split'] # test, train, validation
        bpm = row['bpm'] # 
        midi = row['midi_filename'] #
        f = os.path.join(root, midi)
        if '4-4' not in f: 
            print("skipping", f)
            continue
        tokens = ngt.encode(f)
        if split == 'test':
            test.append(ngt.sequence_division_token)
            for t in tokens: test.append(t)
        elif split == 'train':
            train.append(ngt.sequence_division_token)
            for t in tokens: train.append(t)
        elif split == 'validation':
            val.append(ngt.sequence_division_token)
            for t in tokens: val.append(t)
        else:
            print("ERROR: unknown split type, skipping file")

# print lengths of each split
print(f'test: {len(test)}')
print(f'train: {len(train)}')
print(f'val: {len(val)}')

# save all as npy file 
data = np.array([test, train, val])
np.save('datasets/nuttall_groove_encoded_test_train_val.npy', data, allow_pickle=True)
