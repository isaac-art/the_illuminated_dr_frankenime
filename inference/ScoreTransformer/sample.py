import torch
import torch.nn.functional as F
from pathlib import Path
from miditoolkit import MidiFile
from miditok import MIDILike, TokenizerConfig

from models.papers.lupker_2021 import ScoreTransformer
from utils.inference.transformer_utils import top_k_top_p_filtering


vocab_size = 245
max_seq_len = 256

tokenizer = MIDILike(params=Path(f"datasets/lupker_maestro_midi_{max_seq_len}.json"))

device = 'mps'
model = ScoreTransformer(num_tokens=vocab_size, max_seq_len=max_seq_len).to(device)
model.load_state_dict(torch.load(f'weights/score_transformer_{max_seq_len}_78000.pth', map_location=torch.device(device)))
model.eval()


ARGMAXED = False
temp = 0.9  # Temperature, 1.0 = no change
top_k = 35  # Only consider top_k tokens
top_p = 0.9  # 

target_out_len = 256
input_midi = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/maestro-v3.0.0/2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.midi'
encoded = tokenizer(input_midi)[0].ids[:256] # first n tokens from input midi
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
        # print(generated_tokens[-1])

# add ori to start of generated tokens
generated_tokens = [ori + generated_tokens]
print(generated_tokens)
midi_out = tokenizer(generated_tokens)
midi_out.dump('samples/st/output.mid')