import os
import time
import torch
import tempfile
import gradio as gr
from x_transformers import XLAutoregressiveWrapper

from models.papers.nuttall_2021 import RhythmTransformerXL
from utils.data import NuttallGrooveTokenizer

#PAPER: generation (top K=25, temperature=0.95) or continuation (top K=25, temperature=0.92)

print("starting...")
ngt = NuttallGrooveTokenizer()
device = 'mps'
trained_steps = 1000
max_seq_len = 2048
max_mem_len = 128
model = RhythmTransformerXL(num_tokens=40, max_seq_len=max_seq_len, max_mem_len=max_mem_len).to(device)
model.eval()
print(model)
weights = f"weights/rhythm_transformer_{max_seq_len}_{max_mem_len}_{trained_steps}.pt"
model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
print("loading weights...", weights)
xl_wrapper = XLAutoregressiveWrapper(model)
print("priming model...")


def process_midi(input_midi, temp=0.95, seq_len=512, num_prime_tokens=20):
    start = time.time()
    print(input_midi.name, temp, seq_len, num_prime_tokens)
    encoded = ngt.encode(input_midi.name)
    if int(num_prime_tokens) > len(encoded): return "error num_prime_tokens must be smaller than encoded input length"
    encoded = encoded[:int(num_prime_tokens)]
    encoded_torch = torch.tensor(encoded).unsqueeze(0).to(device)
    print("generating ...")
    gen = xl_wrapper.generate(start_tokens=encoded_torch, seq_len=int(seq_len), eos_token=0, temperature=temp)
    print("decoding...")
    midi_out = ngt.decode(gen[0].cpu().numpy().tolist())
    temp_dir = tempfile.mkdtemp()    
    output_midi_path = os.path.join(temp_dir, 'output.mid')
    print("saving midi...", output_midi_path)
    midi_out.write(output_midi_path)
    print("completed in", time.time()-start, "seconds")
    return output_midi_path

iface = gr.Interface(
    fn=process_midi,
    inputs=[
        gr.File(label="file"),
        gr.Number(label="temp",  value=0.95, minimum=0.1, maximum=1.0),
        gr.Number(label="seq_len", value=512, minimum=32, maximum=3000),
        gr.Number(label="num_prime_tokens", value=20, minimum=1, maximum=512),
    ],
    outputs="text",
    live=False
)

iface.launch()