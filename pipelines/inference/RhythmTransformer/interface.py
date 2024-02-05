import os
import time
import torch
import tempfile
import gradio as gr
from math import ceil
from x_transformers import XLAutoregressiveWrapper
from x_transformers.autoregressive_wrapper import top_p

from models.papers import RhythmTransformerXL
from utils.data import NuttallGrooveTokenizer

#PAPER: generation (top K=25, temperature=0.95) or continuation (top K=25, temperature=0.92)

print("starting...")
ngt = NuttallGrooveTokenizer()
device = 'mps'
vocab = 40
max_seq_len = 96
max_mem_len = 144
model = RhythmTransformerXL(num_tokens=vocab, max_seq_len=max_seq_len, max_mem_len=max_mem_len).to(device)
model.eval()
print(model)
print("-"*80)

weights = f"archive/rhythmtransformer_submission.pt"
model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
print("loading weights...", weights)

def process(input_midi, temp=0.95, seq_len=128, num_prime_tokens=20, include_prime=True):
    try:
        start = time.time()
        xl_wrapper = XLAutoregressiveWrapper(model)
        print(input_midi.name, temp, seq_len, num_prime_tokens)
        encoded = ngt.encode(input_midi.name)
        if int(num_prime_tokens) > len(encoded): return "error num_prime_tokens must be smaller than encoded input length"
        encoded = encoded[:int(num_prime_tokens)]
        encoded_torch = torch.tensor(encoded).unsqueeze(0).to(device)
        print("encoded", len(encoded), "tokens")
        print("generating...")
        gen = xl_wrapper.generate(start_tokens=encoded_torch, seq_len=int(seq_len), eos_token=0, temperature=temp, filter_logits_fn=top_p)
        if include_prime: midi_out = ngt.decode(encoded + gen[0].cpu().numpy().tolist())
        else: midi_out = ngt.decode(gen[0].cpu().numpy().tolist())
        temp_dir = tempfile.mkdtemp()    
        output_midi_path = os.path.join(temp_dir, 'output.mid')
        midi_out.write(output_midi_path)
        print("completed in", time.time()-start, "seconds")
        return output_midi_path
    except Exception as e:
        print(e)
        raise e

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
    """
    # Rhythm Transformer Demo
    """)
    with gr.Row():
        with gr.Column():
            input_midi = gr.File(label="MIDI in")
            with gr.Accordion("Details", open=False):
                temp = gr.Slider(label="temp",  value=0.95, minimum=0.1, maximum=1.0)
                seq_len = gr.Slider(label="seq_len", value=128, minimum=32, maximum=2048)
                num_prime_tokens = gr.Slider(label="num_prime_tokens", value=32, minimum=1, maximum=512)
            include_prime = gr.Checkbox(label="include start seq in output",  value=True)
            process_btn = gr.Button("Process")
        with gr.Column():
            output = gr.File(label="MIDI out")
    process_btn.click(fn=process, inputs=[input_midi, temp, seq_len, num_prime_tokens, include_prime], outputs=output, api_name="process")
iface.launch()
