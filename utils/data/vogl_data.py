import copy
import numpy as np
from miditoolkit import MidiFile, Note, Instrument

class VoglRBMData():
    def __init__(self):
        self.TICKS_PER_MEASURE = 480
        self.TICKS_PER_16TH = 120
        self.DRUMS = 4
        self.STEPS = 16
        self.INPUT_SIZE = self.DRUMS * self.STEPS # 64

    def quantize_notes(self, midi_obj):
        for idx, note in enumerate(midi_obj.instruments[0].notes):
            note.start = round(note.start / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            note.end = round(note.end / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            midi_obj.instruments[0].notes[idx] = note
        return midi_obj
    
    def binary(self, measure):
        # each measure converted into a 64 bit (for the 4 by 16 rhythm patterns) binary vector format
        return

    def encode(self, midi_path):
        midi_obj = MidiFile(midi_path)
        midi_obj = self.quantize_notes(midi_obj)
        
        pass
 