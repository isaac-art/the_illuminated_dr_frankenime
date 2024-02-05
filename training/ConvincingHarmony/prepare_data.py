import os
import numpy as np

from utils import p_
from utils.data import NAESSEncoder #theyre using the same tokenising method
from utils.training import make_deterministic

p_()
make_deterministic()


data = []
data_dir = "datasets/Bach_Chorales/"
tracks = ["Alto.midi", "Bass.midi", "Soprano.midi", "Tenor.midi"]
# PAPER: We consistently paired up soprano with alto, and tenor 
#       with bass, where soprano and tenor were used as training 
#       input and alto and bass as targets.
# ("Soprano", "Alto"),("Tenor", "Bass") . 
inputs = []
targets = []

count = 0
ne = NAESSEncoder()

for root, dirs, files in os.walk(data_dir):
    # more controlled loop thru subdirs too so we know which files are paired
    for d in dirs:
        print(root, d)
        soprano_path = os.path.join(root, d, "Soprano.midi")
        alto_path = os.path.join(root, d, "Alto.midi")
        tenor_path = os.path.join(root, d, "Tenor.midi")
        bass_path = os.path.join(root, d, "Bass.midi")
        if os.path.exists(soprano_path) and os.path.exists(alto_path):
            # we have a pair
            soprano = ne.encode(soprano_path).astype(np.int32)
            alto = ne.encode(alto_path).astype(np.int32)
            inputs.append(soprano)
            targets.append(alto)
        if os.path.exists(tenor_path) and os.path.exists(bass_path):
            # we have a pair
            tenor = ne.encode(tenor_path).astype(np.int32)
            bass = ne.encode(bass_path).astype(np.int32)
            inputs.append(tenor)
            targets.append(bass)

print(len(inputs), len(targets)) #471 471

out_inputs = []
out_targets = []
chunk_size = 32 # PAPER:each sequence was split into shorter sequences of 32 timesteps.
for idx, seq in enumerate(inputs):
    for i in range(0, len(seq), chunk_size):
        input_segment = seq[i:i+chunk_size]
        target_segment = targets[idx][i:i+chunk_size]
        if len(input_segment) == chunk_size:
            out_inputs.append(input_segment)
            if len(target_segment) != chunk_size:
                # pad with 129
                diff = chunk_size - len(target_segment)
                target_segment = np.append(target_segment, [129] * diff)
            out_targets.append(target_segment)
            # print("input", input_segment)
            # print("target", target_segment)
            # print("-"*120)

# create augmentations (reverse, transpose)
# augmented_inputs = []
# augmented_targets = []
# for idx, seq in enumerate(out_inputs):
#     # apply to every 5th seq
#     if idx % 5 == 0:
#         # reverse
#         reversed_input = seq[::-1]
#         reversed_target = out_targets[idx][::-1]
#         augmented_inputs.append(reversed_input)
#         augmented_targets.append(reversed_target)
#         # transpose (change any number not 129 or 128 by +-(0,12))
#         dist = np.random.randint(-12, 13)  # include 12
#         transposed_input = np.where(np.logical_or(seq == 129, seq == 128), seq, seq + dist)
#         transposed_target = np.where(np.logical_or(out_targets[idx] == 129, out_targets[idx] == 128), out_targets[idx], out_targets[idx] + dist)
#         augmented_inputs.append(transposed_input)
#         augmented_targets.append(transposed_target)
# out_inputs = out_inputs + augmented_inputs
# out_targets = out_targets + augmented_targets

out_inputs = np.array(out_inputs)
out_targets = np.array(out_targets)
print(out_inputs.shape, out_targets.shape) #(4423, 32) (4423, 32)- augmented:(6193, 32) (6193, 32)
out_inputs = out_inputs.astype(np.int32)
out_targets = out_targets.astype(np.int32)
np.save("datasets/ch_inputs.npy", out_inputs)
np.save("datasets/ch_targets.npy", out_targets)