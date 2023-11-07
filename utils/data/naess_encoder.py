import numpy as np
from miditoolkit import MidiFile, Note, Instrument
# each subdir has Alto.midi Bass.midi Soprano.midi Tenor.midi

# for each file in each subdir, load the midi file
# tokenize the notes
# The system encodes music using sequences of integers in the 
# range 0–129. 0–127 are pitches from the standard MIDI format, 
# 128 tells the system to stop the note that was playing, and 
# 129 represents no change. Each integer event has a duration 
# of one sixteenth note

# and each voice was used as a single melody during training, 
# giving a total of 1620 training examples of monophonic melodies. 
# All melody lines were transposed to C major and A minor prior to training.

class NAESSEncoder():
    def __init__(self):
        self.TICKS_PER_MEASURE = 480
        self.TICKS_PER_16TH = 120

    def quantize_notes(self, midi_obj):
        for idx, note in enumerate(midi_obj.instruments[0].notes):
            note.start = round(note.start / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            note.end = round(note.end / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            midi_obj.instruments[0].notes[idx] = note
        return midi_obj

    def encode(self, midi_path):
        midi_obj = MidiFile(midi_path)
        midi_obj = self.quantize_notes(midi_obj)
        notes = midi_obj.instruments[0].notes
        notes.sort(key=lambda x: x.start)
        encoded = np.array([])
        for note in notes:
            encoded = np.append(encoded, note.pitch)
            duration = round(note.end - note.start) // (self.TICKS_PER_16TH *20) # (WHY * 20???)
            for _ in range(duration - 1):
                encoded = np.append(encoded, 129)
            encoded = np.append(encoded, 128)
        return encoded

    def decode(self, encoded):
        # encoded should be a sequence of integers
        new_midi = MidiFile()
        new_midi.ticks_per_beat = self.TICKS_PER_MEASURE
        new_midi.instruments.append(Instrument(0, is_drum=False, name="Piano"))
        counter = 0
        for idx, note in enumerate(encoded):
            if note == 128 or note == 129: continue
            note_duration = 0
            for i in range(idx + 1, len(encoded)):
                if encoded[i] == 129: note_duration += 1
                else: break
            note_duration = (note_duration + 1) * self.TICKS_PER_16TH
            start = counter 
            end = start + note_duration
            new_note = Note(
                velocity=100,
                pitch=note,
                start=start,
                end=end
            )
            new_midi.instruments[0].notes.append(new_note)
            counter += note_duration
        return new_midi