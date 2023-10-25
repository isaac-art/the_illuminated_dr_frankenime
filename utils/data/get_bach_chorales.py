import os
from music21 import corpus, midi

def separate_parts_to_midi(chorale, folder_name):
    for part in chorale.parts:
        part_id = part.id
        mf = midi.translate.music21ObjectToMidiFile(part)
        file_path = os.path.join(folder_name, f"{part_id}.midi")
        mf.open(file_path, 'wb')
        mf.write()
        mf.close()

def full_midi(chorale, folder_name):
    mf = midi.translate.music21ObjectToMidiFile(chorale)
    file_path = os.path.join(folder_name, f"full.midi")
    mf.open(file_path, 'wb')
    mf.write()
    mf.close()

if not os.path.exists('datasets/Bach_Chorales'):
    os.makedirs('datasets/Bach_Chorales')

for chorale in corpus.chorales.Iterator():
    folder_name = os.path.join('datasets/Bach_Chorales', f"BWV{chorale.metadata.title}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    separate_parts_to_midi(chorale, folder_name)
    full_midi(chorale, folder_name)