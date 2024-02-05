import copy
from math import floor
import numpy as np
from miditoolkit import MidiFile, Note, Instrument

roland_to_nuttall_pitch_mapping = {
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

class GillickDataMaker:
    def __init__(self):
        self.beats = 32
        self.drum_count = len(set(roland_to_nuttall_pitch_mapping.values())) # 9 - 35min 51max
        self.drum_tokens = {}
        for i, drum in enumerate(set(roland_to_nuttall_pitch_mapping.values())):
            self.drum_tokens[drum] = i

    def map(self, x, min=0, max=127, out_min=0, out_max=1):
        return (x - min) * (out_max - out_min) / (max - min) + out_min
    
    def map_pitch(self, x):
        return self.drum_tokens[roland_to_nuttall_pitch_mapping[x]]

    def quantize_notes(self, midi_obj, ticks_per_16th):
        for idx, note in enumerate(midi_obj.instruments[0].notes):
            note.start = round(note.start / ticks_per_16th) * ticks_per_16th
            note.end = round(note.end / ticks_per_16th) * ticks_per_16th
            midi_obj.instruments[0].notes[idx] = note
        return midi_obj

    def notes(self, x, ticks_per_16th):
        notes_matrix = np.array([[0 for _ in range(self.drum_count)] for _ in range(self.beats)])
        for note in x:
            notes_matrix[int(note.start / ticks_per_16th)][self.map_pitch(note.pitch)] = 1
        return notes_matrix
    
    def velocities(self, x, ticks_per_16th):
        velocity_matrix = np.array([[0.0 for _ in range(self.drum_count)] for _ in range(self.beats)])
        for note in x:
            velocity = self.map(note.velocity)
            velocity_matrix[int(note.start / ticks_per_16th)][self.map_pitch(note.pitch)] = velocity
        return velocity_matrix
    
    def offsets(self, x, ticks_per_16th):
        offset_matrix = np.array([[0.0 for _ in range(self.drum_count)] for _ in range(self.beats)])
        for note in x:
            dist_to_16th = note.start % ticks_per_16th
            offset = self.map(dist_to_16th, min=0, max=ticks_per_16th, out_min=-0.5, out_max=0.5)    
            offset_matrix[int(note.start / ticks_per_16th)][self.map_pitch(note.pitch)] = offset
        return offset_matrix

    def split_measures(self, midi_obj):
        ticks_per_beat = 480 #midi_obj.ticks_per_beat
        beats_per_measure = 4
        measures_per_split = 2 
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

    def encode(self, midi):
        midi_obj = MidiFile(midi)
        ticks_per_beat = 480 #midi_obj.ticks_per_beat
        ticks_per_16th = ticks_per_beat // 4
        quantized = self.quantize_notes(midi_obj, ticks_per_16th)
        q_measures = self.split_measures(quantized)
        o_measures = self.split_measures(midi_obj)
        notes_m = np.array([self.notes(measure, ticks_per_16th) for measure in q_measures])
        velocities_m =  np.array([self.velocities(measure, ticks_per_16th) for measure in o_measures])
        offsets_m =  np.array([self.offsets(measure, ticks_per_16th) for measure in o_measures])
        timings = np.concatenate((velocities_m, offsets_m), axis=2)
        return notes_m, timings
    
    def decode(self, notes=None, timings=None): 
        # print(notes.shape, timings.shape) #(8, 32, 9) (8, 32, 18)
        midi_obj = MidiFile()
        if notes is None and timings is None: return midi_obj # empty midi
        ticks_per_beat = 480
        ticks_per_16th = ticks_per_beat // 4
        midi_obj.ticks_per_beat = ticks_per_beat
        midi_obj.instruments.append(Instrument(program=0, is_drum=True))
        # notes is (measures, beats, drums)
        # timings is (measures, beats, feats*2), where feats is 2 (vel, offset)
        enumerator = enumerate(notes) if notes is not None else enumerate(timings)
        # print("-"*120)
        for i, measure in enumerator:
            for j, beat in enumerate(measure):
                for k, drum in enumerate(beat[:self.drum_count]):
                    if drum != 0: # if there is data 
                        start = i * ticks_per_16th * 16 + j * ticks_per_16th
                        if notes is None: pitch = 36
                        else: pitch = list(self.drum_tokens.keys())[list(self.drum_tokens.values()).index(k)]
                        if timings is None: velocity = 127
                        else: 
                            velocity = max(min(int(self.map(timings[i][j][k], 0, 1, 0, 127)), 127), 0)
                            start += int(self.map(timings[i][j][k+self.drum_count], -0.5, 0.5, 0, ticks_per_16th))
                        end = start + ticks_per_16th
                        note = Note(start=start, end=end, pitch=pitch, velocity=velocity)
                        # print(note)
                        midi_obj.instruments[0].notes.append(note)
        return midi_obj


        # inc = 0
        # if notes_only: skip = 1
        # elif timing_only: skip = 3
        # else: skip = 4
        # midiobj = PrettyMIDI()
        # instrument = Instrument(program=0, is_drum=True)
        # midiobj.instruments.append(instrument)
        # for i in range(0, len(measure), skip):
        #     if timing_only:
        #         start = measure[i] 
        #         end = measure[i+1] 
        #         velocity = measure[i+2]
        #         pitch = 36
        #     elif notes_only:
        #         start = inc 
        #         end = inc 
        #         velocity = 127
        #         pitch = measure[i]
        #         inc += 0.1 # arbitrary tick increase
        #     else:
        #         start = measure[i] 
        #         end = measure[i+1] 
        #         velocity = measure[i+2]
        #         pitch = measure[i+3]
        #     instrument.notes.append(Note(start=start, end=end, pitch=pitch, velocity=velocity))
        # return midiobj
