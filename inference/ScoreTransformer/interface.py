import os
import torch
import shutil
import tempfile
import gradio as gr
from pathlib import Path
import torch.nn.functional as F
from miditoolkit import MidiFile
from miditok import MIDILike, TokenizerConfig

from models.papers.lupker_2021 import ScoreTransformer
from utils.inference.transformer_utils import top_k_top_p_filtering

device = 'mps'
vocab_size = 245
max_seq_len = 256
tokenizer = MIDILike(params=Path(f"datasets/lupker_maestro_midi_{max_seq_len}.json"))
model = ScoreTransformer(num_tokens=vocab_size, max_seq_len=max_seq_len).to(device)
model.load_state_dict(torch.load(f'weights/score_transformer_{max_seq_len}.pth', map_location=torch.device(device))) #_78000
model.eval()

def process_midi(input_midi, temp=0.9, top_k=35, top_p=0.9, seq_len=512, num_prime_tokens=256, include_prime=True):
    print("processing...", input_midi.name)
    print("temp", temp, "top_k", top_k, "top_p", top_p, "seq_len", seq_len, "num_prime_tokens", num_prime_tokens)
    ARGMAXED = False
    midi_obj = MidiFile(input_midi.name)
    encoded = tokenizer(midi_obj)[0].ids[:int(num_prime_tokens)] # first n tokens from input midi
    ori = encoded.copy()
    generated_tokens = []

    with torch.no_grad():
        while len(generated_tokens) < int(seq_len):
            print(len(generated_tokens), end="\r")
            inseq = torch.tensor(encoded).unsqueeze(0).to(device)
            output = model(inseq)
            output = output.squeeze(0)
            
            if not ARGMAXED:
                logits = output[-1, :] / temp
                filtered_logits = top_k_top_p_filtering(logits, top_k=int(top_k), top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
            else:
                next_token = torch.argmax(output[-1, :]).item()
            generated_tokens.append(next_token)
            encoded.append(next_token)
            encoded = encoded[1:]
    
    if include_prime: full_out = [ori + generated_tokens]
    else: full_out = [generated_tokens]
    midi_out = tokenizer(full_out)

    temp_dir = tempfile.mkdtemp()
    output_midi_path = os.path.join(temp_dir, 'output.mid')
    midi_out.dump(output_midi_path)
    
    return output_midi_path


with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
    """
    # ScoreTransformer Demo
    """)
    with gr.Row():
        with gr.Column():
            input_midi = gr.File(label="MIDI in")
            with gr.Accordion("Details", open=False):
                temp = gr.Slider(label="temp",  value=0.95, minimum=0.1, maximum=1.0)
                top_k = gr.Slider(label="top_k",  value=35, minimum=1, maximum=vocab_size)
                top_p = gr.Slider(label="top_p",  value=0.9, minimum=0.1, maximum=1.0)
                seq_len = gr.Slider(label="seq_len", value=512, minimum=32, maximum=3000)
                num_prime_tokens = gr.Slider(label="num_prime_tokens", value=256, minimum=1, maximum=512)
            include_prime = gr.Checkbox(label="include start seq in output",  value=True)
            process_btn = gr.Button("Process")
        with gr.Column():
            output = gr.File(label="MIDI out")
    process_btn.click(fn=process_midi, inputs=[input_midi, temp, top_k, top_p, seq_len, num_prime_tokens, include_prime], outputs=output, api_name="process_midi")
iface.launch()
