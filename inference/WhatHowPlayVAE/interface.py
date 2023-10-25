import os
import time
import torch
import tempfile
import numpy as np
import gradio as gr

from models.papers.gillick_2021 import WhatHowPlayAuxiliaryVAE
from utils.data import GillickDataMaker

device = 'mps'
model = WhatHowPlayAuxiliaryVAE().to(device)
weights = f"weights/whathowplayauxvae_5.pt" 
model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
model.eval()
gdm = GillickDataMaker()

def process(input_groove=None, input_notes=None, mode="sample"):
    with torch.no_grad():
        start = time.time()
        if mode == "sample":
            output = model.sample()
        elif mode == "sample_groove":
            output = model.sample_groove()
        elif mode == "sample_score":
            output = model.sample_score()
        elif mode == "mix":
            _, groove = gdm.encode(input_groove.name)
            notes, _ = gdm.encode(input_notes.name)
            groove_torch = torch.tensor(groove[:32][0], dtype=torch.float32).unsqueeze(0).to(device)
            notes_torch = torch.tensor(notes[:32][0], dtype=torch.float32).unsqueeze(0).to(device)
            print(groove_torch.shape, notes_torch.shape) # torch.Size([32, 18]), torch.Size([32, 9])
            z1 = model.score_vae.encoder(notes_torch)[0]
            z2 = model.groove_vae.encoder(groove_torch)[0]
            joint = torch.cat((z1, z2), dim=2)[0]
            output = model.joint_decoder(joint)
        elif mode == "add_groove":
            # given notes midi, get z1 use random z2
            notes, _ = gdm.encode(input_notes.name)
            notes_torch = torch.tensor(notes[:32][0], dtype=torch.float32).unsqueeze(0).to(device)
            z1 = model.score_vae.encoder(notes_torch)[0]
            z2 = torch.randn(1, 32, 256).to(device)
            joint = torch.cat((z1, z2), dim=2)[0]
            output = model.joint_decoder(joint)
        elif mode == "add_score":
            _, groove = gdm.encode(input_groove.name)
            groove_torch = torch.tensor(groove[:32][0], dtype=torch.float32).unsqueeze(0).to(device)
            z1 = torch.randn(1, 32, 256).to(device)
            z2 = model.groove_vae.encoder(groove_torch)[0]
            joint = torch.cat((z1, z2), dim=2)[0]
            output = model.joint_decoder(joint)
        else:
            return "Unknown mode"
        
        temp_dir = tempfile.mkdtemp()
        outputnp = output.cpu().numpy() 
        if mode in ["sample", "mix", "add_groove", "add_score"]:
            notes = [np.round(np.clip(outputnp[:, :9], 0, 1)).astype(int)] 
            timings = [np.clip(outputnp[:, 9:], -1, 1)]
            midi = gdm.decode(notes=notes, timings=timings)
        elif mode in ["sample_groove"]:
            timings = [np.clip(outputnp, -1, 1)]
            midi = gdm.decode(timings=timings)
        elif mode in ["sample_score"]:
            notes = [np.round(np.clip(outputnp, 0, 1)).astype(int)] 
            midi = gdm.decode(notes=notes)
        output_midi_path = os.path.join(temp_dir, 'output.mid')
        midi.dump(output_midi_path)
        print("completed in", time.time()-start, "seconds")
        return output_midi_path


with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
    """
    # WhatHowPlayAuxVAE 
    """)
    with gr.Row():
        with gr.Column():
            mode = gr.Dropdown(choices=["sample", "sample_groove", "sample_score", "mix", "add_groove", "add_score"], value="sample")
            # duration = gr.Slider(label="duration", value=12, min=1, max=100)
            notes_midi = gr.File(label="MIDI notes track")
            groove_midi = gr.File(label="MIDI groove track")
            process_btn = gr.Button("Process")
        with gr.Column():
            output = gr.File(label="MIDI out")
    process_btn.click(fn=process, inputs=[notes_midi, groove_midi, mode], outputs=output, api_name="process")
iface.launch()
