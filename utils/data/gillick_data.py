import copy
from pretty_midi import PrettyMIDI, Instrument, Note
from miditoolkit import MidiFile
from utils.data import roland_to_nuttall_pitch_mapping

class GillickDataMaker:
    def __init__(self):
        self.beats = 32
        self.drum_count = len(set(roland_to_nuttall_pitch_mapping.values())) # 9 - 35min 51max
        self.drum_tokens = {}
        offset = 3
        for i, drum in enumerate(set(roland_to_nuttall_pitch_mapping.values())):
            self.drum_tokens[drum] = i + offset

    def map_pitch(self, x):
        return self.drum_tokens[roland_to_nuttall_pitch_mapping[x]]

    def quantize_notes(self, midi_obj, ticks_per_16th):
        for idx, note in enumerate(midi_obj.instruments[0].notes):
            note.start = round(note.start / ticks_per_16th) * ticks_per_16th
            note.end = round(note.end / ticks_per_16th) * ticks_per_16th
            midi_obj.instruments[0].notes[idx] = note
        return midi_obj

    def notes_matrix(self, x, ticks_per_16th):
        # F1 (X) is a quantization function that removes all microtiming 
        # and velocity information from a drum loop (keeping only drum score)
        # remove the timing and velocity information
        # x is a pretty_midi.PrettyMIDI object
        # returns txm matrix. t=32(self.measures*self.measure_split), m=9 (self.drum_count drum notes)
        notes_matrix = [[0 for _ in range(self.drum_count)] for _ in range(self.beats)]
        for note in x:
            print(note)
            note.start = round(note.start / ticks_per_16th) * ticks_per_16th
            note.end = round(note.end / ticks_per_16th) * ticks_per_16th
            
    
    def timing_matrix(self, x):
        #  F2 (X) is a “squashing” function keeping performance characteristics 
        # in the form of microtiming and velocity, but discarding the drum score
        # remove the pitch information
        # x is a pretty_midi.PrettyMIDI object
        timing_seq = []
        for note in x.instruments[0].notes:
            timing_seq.append(note.start)
            timing_seq.append(note.end)
            timing_seq.append(note.velocity)
        # print(timing_seq)
        return timing_seq
    
    def tolist(self, x):
        # measure from pretty_midi.PrettyMIDI object to list
        measure = []
        for note in x.instruments[0].notes:
            measure.append(note.start)
            measure.append(note.end)
            measure.append(note.velocity)
            measure.append(note.pitch) #pitch last as easier to match back
        return measure
    
    def split_measures(self, midi_obj):
        ticks_per_beat = midi_obj.ticks_per_beat
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
        # load the midifile and split into measures
        midi_obj = MidiFile(midi)
        ticks_per_beat = midi_obj.ticks_per_beat
        ticks_per_16th = ticks_per_beat // 4
        quantized = self.quantize_notes(midi_obj, ticks_per_16th)
        measures = self.split_measures(quantized)
        print(f'encoded {len(measures)} measures') # 33 measures
        notes_only = [self.notes_matrix(measure, ticks_per_16th) for measure in measures]
        print(f'encoded {len(notes_only)} measures')
        print(notes_only[0])
        timing_only = [self.timing_matrix(measure, ticks_per_16th) for measure in measures]
        exit()
        midi_obj = PrettyMIDI(midi)
        ticks_per_beat = 480
        measures = []
        downbeats = midi_obj.get_downbeats()
        # for n in range(len(downbeats)-1):
        for n in range(0, len(downbeats)-self.measures, self.hop):
            #  "we use 2 measure (or 2 bar) pattern for all reported experimental evaluations, sliding the window with a hop size of 1 measure."
            midiobjcopy = copy.deepcopy(midi_obj)
            midiobjcopy.adjust_times([downbeats[n], downbeats[n + self.measures]], [0, downbeats[n + self.measures] - downbeats[n]])
            measures.append(midiobjcopy)
        print(f'encoded {len(measures)} measures')
        # combis = self.tolist(midi_obj)
        combis = []
        quantizeds = []
        squasheds = []
        for i, measure in enumerate(measures):
            if len(measure.instruments[0].notes) < 1: 
                print("no notes, skipping measure", i)
                continue
            combis.append(self.tolist(measure))
            quantizeds.append(self.quantizer(measure))
            squasheds.append(self.squasher(measure))
            # print(f'{len(combis), len(quantizeds), len(squasheds)} measures')
        # assert len(combis) == len(squasheds) == len(measures)
        return (combis, quantizeds, squasheds)
    
    def decode(self, measure, notes_only=False, timing_only=False):
        # currently only can do one measure decode, as we lose the timing 
        # between measures and cant stich them back together easily
        inc = 0
        if notes_only: skip = 1
        elif timing_only: skip = 3
        else: skip = 4
        midiobj = PrettyMIDI()
        instrument = Instrument(program=0, is_drum=True)
        midiobj.instruments.append(instrument)
        for i in range(0, len(measure), skip):
            if timing_only:
                start = measure[i] 
                end = measure[i+1] 
                velocity = measure[i+2]
                pitch = 36
            elif notes_only:
                start = inc 
                end = inc 
                velocity = 127
                pitch = measure[i]
                inc += 0.1 # arbitrary tick increase
            else:
                start = measure[i] 
                end = measure[i+1] 
                velocity = measure[i+2]
                pitch = measure[i+3]
            instrument.notes.append(Note(start=start, end=end, pitch=pitch, velocity=velocity))
        return midiobj
