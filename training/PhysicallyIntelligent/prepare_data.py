import os
import numpy as np

from utils.data import NAESSEncoder
# test = '/Users/isaac/Desktop/one/datasets/Bach_Chorales/BWVWir Christenleut’/Bass.midi'
# ne = NAESSEncoder()
# encoded = ne.encode(test)
# decoded = ne.decode(encoded)
# decoded.dump("samples/pii/sample.mid")

data = []
data_dir = "datasets/Bach_Chorales/"
tracks = ["Alto.midi", "Bass.midi", "Soprano.midi", "Tenor.midi"]
count = 0
ne = NAESSEncoder()
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file in tracks:
            count += 1
            print(os.path.join(root, file))
            encoding = ne.encode(os.path.join(root, file))
            data.append(encoding)
print("file count", count) # 943

# get the longest sequence, and pad the rest with 129 to match
# max_len = max([len(x) for x in data])
# print(max_len)
# for idx, seq in enumerate(data):
#     if len(seq) < max_len:
#         diff = max_len - len(seq)  
#         data[idx] = np.append(seq, [129] * diff)

# instead we will chunk all sequences to 128 length
out_data = []
chunk_size = 128
for idx, seq in enumerate(data):
    for i in range(0, len(seq), chunk_size):
        segment = seq[i:i+chunk_size]
        if len(segment) == chunk_size:
            out_data.append(seq[i:i+chunk_size])

out_data = np.array(out_data)
print(out_data.shape) # (943, 1441) #not 1620 melodies described in paper?
out_data = out_data.astype(np.float32)
np.save("datasets/pii.npy", out_data)
