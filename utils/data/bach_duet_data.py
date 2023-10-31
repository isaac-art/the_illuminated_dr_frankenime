import numpy as np
from miditoolkit import MidiFile, Note, Instrument

class BachDuetData():
    def __init__(self):
        self.TICKS_PER_16TH = 480 // 4  # Assuming 480 ticks per beat
        self.TICKS_PER_MEASURE = 4 * 480  # Assuming 4 beats per measure in 4/4 time signature
        self.TICKS_PER_BEAT = 480
        self.RHYTHMS = { (1,  0,  0): 0, (0, -2, -3): 1, (0, -1, -2): 2, (0,  0, -2): 3,
            (0, -2, -4): 4, (0, -1, -3): 5,  (0,  0, -1): 6, (0,  0, -2): 7, (0, -2, -4): 8}
    
    def quantize_notes(self, midi_obj):
        for idx, note in enumerate(midi_obj.instruments[0].notes):
            note.start = round(note.start / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            note.end = round(note.end / self.TICKS_PER_16TH) * self.TICKS_PER_16TH
            midi_obj.instruments[0].notes[idx] = note
        return midi_obj
     
    def enc_midi_artic(self, pitch, is_start):
        pitch = min(max(pitch, 28), 94) #clamp input
        m = int(67 * ((pitch - 28) / (94 - 28))) #map to token range
        if is_start: m *= 2 #double for start of note
        return m
    
    def dec_midi_artic(self, m):
        is_start = False
        if m > 67: m //= 2; is_start = True
        pitch = int(28 + ((m / 67) * ((94+1) - 28))) # had to +1 to get correct note, rounding ?
        return pitch, is_start  
    
    def enc_rhythm(self, frame_idx):
        bar_pattern =    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        beat_pattern =   [0, -2, -1, -2,  0, -2, -1, -2,  0, -2, -1, -2,  0, -2, -1, -2]
        accent_pattern = [0, -3, -2, -3, -2, -4, -3, -4, -1, -3, -2, -3, -2, -4, -3, -4]
        bar = bar_pattern[frame_idx % 16]
        beat = beat_pattern[frame_idx % 16]
        accent = accent_pattern[frame_idx % 16]
        return self.RHYTHMS[(bar, beat, accent)]
    
    def dec_rhythm(self, r):
        return list(self.RHYTHMS.keys())[r]
        
    def encode(self, midi_path):
        midi_obj = MidiFile(midi_path)
        self.quantize_notes(midi_obj)
        end_tick = midi_obj.instruments[0].notes[-1].end
        num_frames = end_tick // self.TICKS_PER_16TH
        encoded_midi = np.zeros((num_frames, 3)) # [bar, beat, accent] CPC, midi_artic
        for tick in range(0, end_tick, self.TICKS_PER_16TH):
            frame_idx = tick // self.TICKS_PER_16TH 
            encoded_midi[frame_idx, 0] = self.enc_rhythm(frame_idx)
            encoded_midi[frame_idx, 1] = 12 # CPC set to 12 (rest)
            encoded_midi[frame_idx, 2] = 135 # MIDI Artic set to 135 (rest)
            for note in midi_obj.instruments[0].notes:
                if note.start <= tick and note.end >= tick:
                    # print(note)
                    encoded_midi[frame_idx, 1] = note.pitch % 12  # CPC updated to pitch % 12
                    s = note.start == tick
                    encoded_midi[frame_idx, 2] = self.enc_midi_artic(note.pitch, s) # MIDI Artic updated to pitch_on/hold
                    # break # only one note at a time 
        return encoded_midi
    
    def decode(self, encoded_midi):
        midi_obj = MidiFile()
        midi_obj.ticks_per_beat = self.TICKS_PER_BEAT
        midi_obj.instruments.append(Instrument(program=0, is_drum=False, name='Piano'))
        for frame_idx, frame in enumerate(encoded_midi):
            tick = frame_idx * self.TICKS_PER_16TH // 20
            r, cpc, midi_artic = frame 
            pitch, is_start = self.dec_midi_artic(int(midi_artic))
            # print(frame_idx, midi_artic, pitch, is_start)
            if pitch == 0: continue # skip rests
            if is_start:
                note = Note(start=tick, end=tick, pitch=pitch, velocity=100)
                midi_obj.instruments[0].notes.append(note)
                # print("add", midi_obj.instruments[0].notes[-1])
            else:
                midi_obj.instruments[0].notes[-1].end = tick
                # print("update", midi_obj.instruments[0].notes[-1])
        # print(len(midi_obj.instruments[0].notes)) # we have corrent number of notes, just wrong timing.
        return midi_obj

if __name__ == "__main__":
    midif = "/Users/isaac/Desktop/one/samples/bd/input.midi"
    encoder = BachDuetData()
    encoding = encoder.encode(midif)
    # print(encoding[100:300])
    decoding = encoder.decode(encoding)
    decoding.dump("samples/bd/remake.mid")