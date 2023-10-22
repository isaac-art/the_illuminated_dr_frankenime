import os
import csv
import torch
import pretty_midi
import numpy as np
from pathlib import Path
from collections import deque
from miditoolkit import MidiFile, Note



class NuttallEncoder():
    def __init__(self):
        self.quantize = 16 # 16th notes
        self.roland_to_nuttall_pitch_mapping = {
            36: 35, #kick-bass
            38: 38, #snare(head)-snare
            40: 38, #snare(rim)-snare
            37: 38, #snare(xstick)-snare
            48: 50, #tom1-high tom
            50: 50, #tom1-high tom
            45: 48, #tom2-lowmid tom
            47: 48, #tom2-lowmid tom
            43: 45, #tom3-highfloor tom
            58: 45, #tom3-highfloor tom
            46: 46, #hhopen(bow)-hhopen
            26: 46, #hhopen(edge)-hhopen
            42: 42, #hhclosed(bow)-hhclosed
            22: 42, #hhclosed(edge)-hhclosed
            44: 42, #hhpedal(bow)-hhclosed
            49: 49, #crash1(bow)-crash
            55: 49, #crash1(edge)-crash
            57: 49, #crash2(bow)-crash
            52: 49, #crash2(edge)-crash
            51: 51, #ride(bow)-ride
            59: 51, #ride(edge)-ride
            53: 51, #ride(bell)-ride
        }
        self.time_tokens_ticks = {
                1: 1, # 1 tick
                2: 10, # 10 ticks
                3: 100, # 100 ticks
                4: 1000, # 1000 ticks
                5: 10000, # 10000 ticks
            }
        self.pitch_list = set(self.roland_to_nuttall_pitch_mapping.values())
        self.num_pitches = len(self.pitch_list) #should be 9
        self.num_v_buckets = 4 # 4 velocity buckets
        self.num_pitch_vel_tokens = self.num_pitches * self.num_v_buckets # 36
        self.token_pitch_vel_pairs = self.pitch_velocity_tokens(self.pitch_list, self.num_v_buckets)

    def pitch_velocity_tokens(self, pitch_list, num_v_buckets=4):
        token_pitch_vel_pairs = {}
        counter = 0
        for pitch in pitch_list:
            for v in range(1, num_v_buckets + 1):
                token_pitch_vel_pairs[counter] = (pitch, v)
                counter += 1 
        return token_pitch_vel_pairs

    def velocity_to_bucket(self, velocity):
        if velocity < 64: return 1
        elif velocity < 96: return 2
        elif velocity < 128: return 3
        return 4

    def encode(self, f):
        midi_obj = MidiFile(f)
        # all sequences are first quantized to 16th notes
        ticks_per_beat = midi_obj.ticks_per_beat
        ticks_per_16th = ticks_per_beat // 4
        for i, instrument in enumerate(midi_obj.instruments):
            for j, note in enumerate(instrument.notes):
                # Quantize note start and end times to the nearest 16th note
                note.start = round(note.start / ticks_per_16th) * ticks_per_16th
                note.end = round(note.end / ticks_per_16th) * ticks_per_16th
        # then pitches mapped to nuttall mapping
        for instrument in midi_obj.instruments:
            for note in instrument.notes:
                note.pitch = self.roland_to_nuttall_pitch_mapping.get(note.pitch, note.pitch)
        # then velcoities are mapped to 4 buckets
        for instrument in midi_obj.instruments:
            for note in instrument.notes:
                note.velocity = self.velocity_to_bucket(note.velocity)
        # then 

    def decode(self, tokens):
        pass


# train 378, test 77, validation 48 split
# approx~ 75% train, 15% test, 10% validation
encoder = NuttallEncoder()
exit()
test, train, val = [], [], []
root = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'
groove_data =  f'{root}/info.csv'
with open(groove_data, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)
    for row in data:
        split = row['split'] # test, train, validation
        bpm = row['bpm'] # 
        midi = row['midi_filename'] #
        f = os.path.join(root, midi)
        tokens = encoder.encode(f)


# Finally, every (pitch mn , velocity bucket bn ) combination is assigned a 
# unique token corresponding to that pair. With B = 4 and 9 pitch classes, 
# we thus have 36 (9 × 4) unique tokens, corresponding to every possible 
# combination of (mn , bn )

# seq = [(pv_token1, time_token1), (pv_token2, time_token2), ... (pv_tokenN, time_tokenN)]
time_tokens_ticks = {
    1: 1, # 1 tick
    2: 10, # 10 ticks
    3: 100, # 100 ticks
    4: 1000, # 1000 ticks
    5: 10000, # 10000 ticks
}
# Silences are filled with as few individual tick tokens as possible for the duration. 
# For example a silence of 345 ticks is represented by [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1] 
# (3 × one hundred tokens, 4 × ten tokens and 5 × one tokens). Similarly, a silence of 
# 5003 ticks would be represented by [4, 4, 4, 4, 4, 1, 1, 1].

# then all sequences are concatenated into one long stream with a division token


exit()
NuttallConfig = {
    'nb_velocities': 4,
    'nb_tempos': 1,
    'use_chords': False,
    'use_programs': False,
    'use_tempos': False,
    'use_time_signatures': False,
    'one_token_stream': True,
}

# MIDILike is closest to This time with feeling (Oore et al.) 
config = TokenizerConfig(**NuttallConfig)
tokenizer = MIDILike(config)

maestro_dir = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'

seq_len = 2048
tokenised_stream = []
file_count = 0
for root, dirs, files in os.walk(maestro_dir):
    for file in files:
        if file.endswith('mid') or file.endswith('midi'):
            file_count += 1
            print(file_count, end="\r")
            midi = MidiFile(os.path.join(root, file))
            tokens = tokenizer(midi)
            # chunk into multiple sequences of len seq_len (discard remainder)
            for i in range(0, len(tokens[0].ids), seq_len):
                s = tokens[0].ids[i:i+seq_len]
                if len(s) == seq_len:
                    # print(len(s))
                    # tokenised_stream += DIVISION TOKEN HERE
                    tokenised_stream += tokens[0].ids[i:i+seq_len]
            # tokenised.append(tokens[0].ids)

# shoudl be one long stream

tokenizer.save_params(f'datasets/nuttall_groove_midi_{seq_len}.json')
np.save(f'datasets/nuttall_groove_midi_{seq_len}.npy', tokenised, allow_pickle=True)

#max([max(seq) for seq in self.tokenized_sequences]) + 1
print("vocab size:", max([max(seq) for seq in tokenised]) + 1)