import copy
from pretty_midi import PrettyMIDI, Instrument, Note

class GillickDataMaker:
    def __init__(self):
        self.measures = 1
        self.beats_per_measure = 4

    def quantizer(self, x):
        # F1 (X) is a quantization function that removes all microtiming 
        # and velocity information from a drum loop (keeping only drum score)
        # remove the timing and velocity information
        # x is a pretty_midi.PrettyMIDI object
        pitch_seq = [note.pitch for note in x.instruments[0].notes]
        return pitch_seq
    
    def squasher(self, x):
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
            measure.append(note.pitch)
            measure.append(note.velocity)
        return measure
    
    def encode(self, midi):
        # load the midifile and split into measures
        midi_obj = PrettyMIDI(midi)
        ticks_per_beat = 480
        ticks_per_measure = ticks_per_beat * self.beats_per_measure 
        ticks_per_beat_to_time = midi_obj.tick_to_time(ticks_per_beat)
        ticks_per_measure_to_time = midi_obj.tick_to_time(ticks_per_measure)
        measures = []
        downbeats = midi_obj.get_downbeats()
        # for n in range(len(downbeats)-1):
        for n in range(len(downbeats)-1):
            midiobjcopy = copy.deepcopy(midi_obj)
            midiobjcopy.adjust_times([downbeats[n], downbeats[n + 1]], [0, downbeats[n + 1] - downbeats[n]])
            measures.append(midiobjcopy)
        print(f'encoded {len(measures)} measures')
        # combis = self.tolist(midi_obj)
        combis = []
        quantizeds = []
        squasheds = []
        for measure in measures:
            if len(measure.instruments[0].notes) < 1: 
                print("no notes, skipping measure", len(measure.instruments[0].notes))
                continue
            combis.append(self.tolist(measure))
            quantizeds.append(self.quantizer(measure))
            squasheds.append(self.squasher(measure))
        assert len(combis) == len(squasheds) == len(measures)
        return (combis, quantizeds, squasheds)
    
    def decode(self, measure, notes_only=False, timing_only=False):
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
                pitch = 36
                velocity = measure[i+2]
            elif notes_only:
                start = inc 
                end = inc 
                pitch = measure[i]
                velocity = 127
                inc += 0.1 # arbitrary tick increase
            else:
                start = measure[i] 
                end = measure[i+1] 
                pitch = measure[i+2]
                velocity = measure[i+3]
            instrument.notes.append(Note(start=start, end=end, pitch=pitch, velocity=velocity))
        return midiobj
