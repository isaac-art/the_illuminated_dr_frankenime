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
model.load_state_dict(torch.load(f'weights/score_transformer_{max_seq_len}_25500.pth', map_location=torch.device(device)))
model.eval()

def process_midi(input_midi, temp=0.9, top_k=35, top_p=0.9, target_out_len=512, input_limit=256):
    print("processing...", input_midi.name)
    ARGMAXED = False
    midi_obj = MidiFile(input_midi.name)
    encoded = tokenizer(midi_obj)[0].ids[:input_limit] # first n tokens from input midi
    ori = encoded.copy()
    generated_tokens = []

    with torch.no_grad():
        while len(generated_tokens) < target_out_len:
            print(len(generated_tokens), end="\r")
            inseq = torch.tensor(encoded).unsqueeze(0).to(device)
            output = model(inseq)
            output = output.squeeze(0)
            
            if not ARGMAXED:
                logits = output[-1, :] / temp
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
            else:
                next_token = torch.argmax(output[-1, :]).item()
            generated_tokens.append(next_token)
            encoded.append(next_token)
            encoded = encoded[1:]
    
    full_out = [ori + generated_tokens]
    midi_out = tokenizer(full_out)

    temp_dir = tempfile.mkdtemp()
    output_midi_path = os.path.join(temp_dir, 'output.mid')
    midi_out.dump(output_midi_path)
    
    return output_midi_path

iface = gr.Interface(
    fn=process_midi,
    inputs=[
        "file",
        gr.inputs.Number(default=1.0, label="Temperature"),
        gr.inputs.Number(default=35, label="Top K"),
        gr.inputs.Number(default=0.9, label="Top P"),
        gr.inputs.Number(default=512, label="Target Output Length"),
        gr.inputs.Number(default=256, label="Input Limit")
    ],
    outputs="text",
    live=False
)

iface.launch()