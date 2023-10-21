import torch
from miditoolkit import MidiFile
from miditok import MIDILike, TokenizerConfig

from models.papers.lupker_2021 import ScoreTransformer

LupkerConfig = {
    'nb_velocities': 1,
    # 'beat_res': {(0,-1):16},
    'nb_tempos': 1,
    'use_chords': False,
    'use_programs': False,
    'use_tempos': False,
    'use_time_signatures': False,
}

config = TokenizerConfig(**LupkerConfig)
tokenizer = MIDILike(config)

model = ScoreTransformer()
model.load_state_dict(torch.load('weights/scoretransformer.pth', map_location=torch.device('cpu')))
model.eval()


input_midi = input("Input MIDI file path: ")
encoded = tokenizer(input_midi)[0].ids
output = model(torch.tensor(encoded).unsqueeze(0), None)
output = torch.argmax(output, dim=2).squeeze(0).tolist()
midi_out = tokenizer(output)
midi_out.save('output.mid')