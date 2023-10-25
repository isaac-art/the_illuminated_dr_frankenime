import os
import time
import torch
import tempfile
import numpy as np
import gradio as gr

from models.papers.gillick_2021 import WhatHowPlayAuxiliaryVAE
from utils.data import GillickDataMaker

device = 'mps'
gdm = GillickDataMaker()

with torch.no_grad():
    model = WhatHowPlayAuxiliaryVAE().to(device)
    weights = f"weights/whathowplayauxvae_2.pt"
    model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    model.eval()

    output = model.sample()
    outputnp = output.cpu().numpy() 
    notes = [np.round(np.clip(outputnp[:, :9], 0, 1)).astype(int)]# make 0-1
    timings = [np.clip(outputnp[:, 9:], -1, 1)]
    midi = gdm.decode(notes=notes, timings=timings)
    temp_dir = tempfile.mkdtemp()
    output_midi_path = os.path.join(temp_dir, 'output.mid')
    midi.dump(output_midi_path)
    print(output_midi_path)

