
from miditoolkit import MidiFile, Note, Instrument


def join_midi(miditracks):
    midi_obj = MidiFile()
    midi_obj.ticks_per_beat = 480
    midi_obj.instruments.append(Instrument(program=0, is_drum=False, name='Piano'))
    midi_input_objs = [MidiFile(miditrack) for miditrack in miditracks]
    for track in midi_input_objs:
        for note in track.instruments[0].notes:
            midi_obj.instruments[0].notes.append(note)
    return midi_obj