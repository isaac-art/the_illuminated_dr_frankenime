import torch
import torch.nn.functional as F
from pathlib import Path
from miditoolkit import MidiFile
from miditok import MIDILike, TokenizerConfig

from models.papers.lupker_2021 import ScoreTransformer


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
       Args:
            logits: logits distribution shape (vocabulary size)
            top_k: top-k tokens to keep, <=0 means no restriction
            top_p: keep only a subset S of candidates, s.t. the sum of their probabilities is >= top_p
       Return:
            filtered_logits: the filtered tensor based on top-k and/or nucleus filtering
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

tokenizer = MIDILike(params=Path("datasets/lupker_maestro_midi.json"))

vocab_size = 245
max_seq_len = 1024
device = 'mps'
model = ScoreTransformer(num_tokens=vocab_size, max_seq_len=max_seq_len).to(device)
model.load_state_dict(torch.load('weights/score_transformer.pth', map_location=torch.device(device)))
model.eval()


temp = 1.0  # Temperature
top_k = 40  # Only consider top_k tokens
top_p = 0.9  # Nucleus sampling parameter

target_out_len = 256
input_midi = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.midi'
encoded = tokenizer(input_midi)[0].ids[:30] # first n tokens from input midi
ori = encoded.copy()
generated_tokens = []

while len(generated_tokens) < target_out_len:
    print(len(generated_tokens), end="\r")
    inseq = torch.tensor(encoded).unsqueeze(0).to(device)
    output = model(inseq)
    output = output.squeeze(0)
    
    # Get the logits and apply temperature
    logits = output[-1, :] / temp
    
    # Apply top-k and/or top-p filtering
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    # Sample from the filtered distribution
    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1).item()
    
    generated_tokens.append(next_token)
    encoded.append(next_token)
    encoded = encoded[1:]

    print(generated_tokens)


# add ori to start of generated tokens
generated_tokens = [ori + generated_tokens]
print(generated_tokens)
midi_out = tokenizer(generated_tokens)
midi_out.dump('samples/st/output.mid')