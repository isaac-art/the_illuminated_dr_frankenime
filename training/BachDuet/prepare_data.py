import os
import numpy as np

from utils import p_
from utils.training import make_deterministic
from utils.data import BachDuetData

p_()
make_deterministic()

bdd = BachDuetData()

data_dir = "datasets/Bach_Chorales/"
tracks = [ "Soprano.midi", "Bass.midi"] # soprano inputs, bass targets

count = 0
inputs = []
targets = []
chunk_size = 32

for root, dirs, files in os.walk(data_dir):
    for d in dirs:
        print(root, d)
        soprano_path = os.path.join(root, d, "Soprano.midi")
        bass_path = os.path.join(root, d, "Bass.midi")
        if os.path.exists(soprano_path) and os.path.exists(bass_path):
            # we have a pair
            soprano = bdd.encode(soprano_path)
            bass = bdd.encode(bass_path)
            
            soprano = soprano[:-(soprano.shape[0] % chunk_size)]
            bass = bass[:-(bass.shape[0] % chunk_size)]

            soprano = soprano.reshape(-1, chunk_size, 3)
            bass = bass.reshape(-1, chunk_size, 3)
            # add all the chunks to the dataset
            for i in range(soprano.shape[0]):
                inputs.append(soprano[i])
                targets.append(bass[i])


out_inputs = np.array(inputs)#.astype(np.int32)
out_targets = np.array(targets)#.astype(np.int32)
np.save("datasets/bdd_inputs_32.npy", out_inputs)
np.save("datasets/bdd_targets_32.npy", out_targets)