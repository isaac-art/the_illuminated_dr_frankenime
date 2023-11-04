import os
import numpy as np
from utils.data import VoglRBMData

drbd = VoglRBMData()
mp = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'
encodings = np.array([])

for root, dirs, files in os.walk(mp):
    for file in files:
        if file.endswith('.mid') or file.endswith('.midi'):
            if '4-4' in file:
                print(file)
                path = os.path.join(root, file)
                encoding = drbd.encode(path)
                encoding_np = np.array(encoding)
                encoding_np = encoding_np.reshape((-1, 64))
                encodings = np.append(encodings, encoding_np)

print(encodings.shape) #(1396096,)
encodings = encodings.reshape((-1, 64))
print(encodings.shape) #21814, 64)
# print(encodings[0])
# save 
np.save('datasets/vogl_encodings.npy', encodings)