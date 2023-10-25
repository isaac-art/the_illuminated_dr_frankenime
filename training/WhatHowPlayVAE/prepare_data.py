# One barrier that has inhibited music generation models from being 
# put to use in the same way as gesture mapping, however, is the size 
# of datasets and expense of computational resources needed to train them, 
# which prevents users from choosing and manipulating their own training data. 

# chunk into 1-measure drum loops
    # split into 
        # 1) drum score and quantized
        # 2) tapped groove and squashed
    # so one side holds timing and the other notes
import os
import pickle

from utils.data import GillickDataMaker


if __name__ == "__main__":
    midi = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/drummer1/eval_session/6_hiphop-groove6_87_beat_4-4.mid'
    
    gdm = GillickDataMaker()
    largest_start = 0
    largest_end = 0
    largest_note = 0
    larget_vel = 0
    mididir = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'
    for root, dirs, files in os.walk(mididir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                combis, quantizeds, squasheds = gdm.encode(os.path.join(root, file))
                print(len(combis), len(quantizeds), len(squasheds), "measures")
                
                for i, measure in enumerate(combis):
                    for j, note in enumerate(measure):
                        # 1st note is start, 2nd is end, 3rd is velocity, 4th is pitch, 5th is start, 6th is end, etc
                        if j % 4 == 0:
                            if note > largest_start: largest_start = note
                        if j % 4 == 1:
                            if note > largest_end: largest_end = note
                        if j % 4 == 2:
                            if note > larget_vel: larget_vel = note
                        if j % 4 == 3:
                            if note > largest_note: largest_note = note
    print("largest start", largest_start)
    print("largest end", largest_end)
    print("largest vel", larget_vel)
    print("largest note", largest_note)

    exit()
    dataset = {
        'combis': [],
        'quantizeds': [],
        'squasheds': []
    }
    mididir = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'
    for root, dirs, files in os.walk(mididir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                combis, quantizeds, squasheds = gdm.encode(os.path.join(root, file))
                if len(combis) < 1: 
                    print("no notes, wont add measure to dataset")
                    continue

                print(f"adding {len(combis), len(quantizeds), len(squasheds)} measures to dataset")
                dataset['combis'].extend(combis)
                dataset['quantizeds'].extend(quantizeds)
                dataset['squasheds'].extend(squasheds)
                print(len(dataset['combis']), len(dataset['quantizeds']), len(dataset['squasheds']))
    
    # save the dataset to /datasets/gillick.pkl
    with open('datasets/gillick.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    midiout = gdm.decode(dataset['combis'][0])
    midiout.write('samples/whp/remake.mid')
    midiout = gdm.decode(dataset['quantizeds'][0], notes_only=True)
    midiout.write('samples/whp/notes.mid')
    midiout = gdm.decode(dataset['squasheds'][0], timing_only=True)
    midiout.write('samples/whp/groove.mid')