import copy
import numpy as np
from miditoolkit import MidiFile, Note, Instrument


roland_to_vogl_pitch_mapping = {
    36: 1, #kick-bass
    38: 2, #snare(head)-snare
    40: 2, #snare(rim)-snare
    37: 2, #snare(xstick)-snare
    46: 3, #hhopen(bow)-hhopen
    26: 3, #hhopen(edge)-hhopen
    42: 4, #hhclosed(bow)-hhclosed
    22: 4, #hhclosed(edge)-hhclosed
    44: 4, #hhpedal(bow)-hhclosed
}

class VoglRBMData():
    def __init__(self):
        self.TICKS_PER_MEASURE = 480
        self.TICKS_PER_16TH = 120
        self.DRUMS = 4
        self.STEPS = 16
        # self.INPUT_SIZE = self.DRUMS * self.STEPS # 64

    def quantize_notes(self, midi_obj):
        for idx, note in enumerate(midi_obj.instruments[0].notes):
            note.start = round(note.start / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            note.end = round(note.end / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            midi_obj.instruments[0].notes[idx] = note
        return midi_obj
    
    def split_measures(self, midi_obj):
        ticks_per_beat = 480 #midi_obj.ticks_per_beat
        beats_per_measure = 4
        measures_per_split = 1
        division = ticks_per_beat * beats_per_measure * measures_per_split
        measures = []
        current = []
        last_tick = 0
        for note in midi_obj.instruments[0].notes:
            while note.start >= last_tick + division:
                measures.append(current)
                current = []
                last_tick += division
            note_copy = copy.deepcopy(note)
            note_copy.start -= last_tick
            note_copy.end -= last_tick
            current.append(note_copy)
        if current: measures.append(current) # append last measure
        return measures
    
    def to_binary(self, measure):
        binary_matrix = np.zeros((self.DRUMS, self.STEPS))
        for note in measure:
            if note.pitch in roland_to_vogl_pitch_mapping.keys():
                mapping = roland_to_vogl_pitch_mapping[note.pitch]
                row = mapping - 1
                start = note.start // self.TICKS_PER_16TH
                binary_matrix[row, start] = 1
        binary_vec = binary_matrix.flatten()
        return binary_vec

    def from_binary(self, binary):
        binary_matrix = binary.reshape((self.DRUMS, self.STEPS))
        measure = []
        for row in range(self.DRUMS):
            for col in range(self.STEPS):
                if binary_matrix[row, col] == 1:
                    pitch = list(roland_to_vogl_pitch_mapping.keys())[row]
                    start = col * self.TICKS_PER_16TH
                    end = start + self.TICKS_PER_16TH
                    note = Note(
                        velocity=100,
                        pitch=pitch,
                        start=start,
                        end=end
                    )
                    measure.append(note)
        return measure

    def encode(self, midi_path):
        midi_obj = MidiFile(midi_path)
        midi_obj = self.quantize_notes(midi_obj)
        measures = self.split_measures(midi_obj)
        binary_measures = []
        for i, measure in enumerate(measures):
            binary = self.to_binary(measure)
            binary_measures.append(binary)
        return binary_measures
    
    def decode(self, binary_measures):
        midi_obj = MidiFile()
        midi_obj.ticks_per_beat = 480
        multiplier = self.TICKS_PER_MEASURE * 4 #120
        midi_obj.instruments.append(Instrument(0, is_drum=True, name='Drums'))
        for i, binary in enumerate(binary_measures):
            measure = self.from_binary(binary)
            for note in measure:
                note.start += i * multiplier
                note.end += i * multiplier
                midi_obj.instruments[0].notes.append(note)
        return midi_obj

if __name__ == "__main__":
    vrd = VoglRBMData()
    mp = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/drummer1/session3/7_dance-disco_120_beat_4-4.mid'
    encoding = vrd.encode(mp)
    # print(encoding)
    decoding = vrd.decode(encoding)
    decoding.dump('samples/drbm/decoded.mid')