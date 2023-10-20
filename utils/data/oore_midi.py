# OORE proposes:
# A MIDI excerpt is represented as a sequence of events from the following vocabulary of 413 different events:
# • 128 NOTE-ON events: one for each of the 128 MIDI pitches. Each one starts a new note.
# • 128 NOTE-OFF events: one for each of the 128 MIDI pitches. Each one releases a note.
# • 125 TIME-SHIFT events: each one moves the time step forward by increments of 8 ms up to 1 second.
# • 32 VELOCITY events: each one changes the velocity applied to all subsequent notes (until the next velocity event).
# LUPKER DIFFERS: vocab_size = 360  # 256 note events + 1 velocity + 100 time-shifts + 3 special tokens

import pretty_midi
import numpy as np

class OoreMIDIEncoder():
    def __init__(self, notes=(128, 128), velocities=32, time_shifts=125, time_spacings_ms=8, special_tokens=["start", "end", "pad"], max_len=2048):
        self.notes = notes
        self.velocities = velocities
        self.time_shifts = time_shifts
        self.time_spacings_ms = time_spacings_ms
        self.special_tokens = special_tokens 
        self.vocab_size = notes[0] + notes[1] + velocities + time_shifts + len(special_tokens) # 360
        self.max_len = max_len
        print('OoreMIDI initialized with vocab_size: ', self.vocab_size)

    def split_into_max_len(self, sequence):
        split_len = self.max_len - 2  # Leaving space for start and end tokens
        return [sequence[i:i + split_len] for i in range(0, len(sequence), split_len)]

    def add_special_tokens(self, sequences):
        if "start" in self.special_tokens:
            sequences = [[self.vocab_size - 3] + seq for seq in sequences]
        if "end" in self.special_tokens:
            sequences = [seq + [self.vocab_size - 2] for seq in sequences]
        if "pad" in self.special_tokens:
            pad_token = self.vocab_size - 1
            sequences = [seq + [pad_token] * (self.max_len - len(seq)) for seq in sequences]
        return sequences

    def compute_time_shift(self, note_start, current_time):
        time_shift = int((note_start - current_time) * 1000 / self.time_spacings_ms) 
        time_shifts = []
        while time_shift >= self.time_shifts:
            time_shifts.append(self.time_shifts)
            time_shift -= self.time_shifts
        if time_shift > 0:
            time_shifts.append(time_shift)
        return time_shifts
    
    def encode(self, midi_file, split_max_len=True):
        pm = pretty_midi.PrettyMIDI(midi_file)
        encoded_sequence = []
        current_time = 0

        for instrument in pm.instruments:
            for note in instrument.notes:
                time_shifts = self.compute_time_shift(note.start, current_time)
                encoded_sequence.extend(time_shifts)
                current_time = note.start
                encoded_sequence.append(note.pitch)
                current_time = note.end
                encoded_sequence.append(self.notes[0] + note.pitch)  # Note OFF encoding
        if split_max_len:
            encoded_sequences = self.split_into_max_len(encoded_sequence)
        encoded_sequences = self.add_special_tokens(encoded_sequences)
        return True, encoded_sequences
    

def main(midi_file, lupker_config=True):
    LupkerConfig = {
        'notes': (128, 128), #on, off
        'velocities': 1,
        'time_shifts': 100,
        'time_spacings_ms': 10,
        'special_tokens': ["start", "end", "pad"],
        'max_len': 2048
    }
    if lupker_config:
        processor = OoreMIDIEncoder(**LupkerConfig)
    else:
        processor = OoreMIDIEncoder()
    encoded_sequences = processor.encode(midi_file)

if __name__ == '__main__':
    main( '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi')
