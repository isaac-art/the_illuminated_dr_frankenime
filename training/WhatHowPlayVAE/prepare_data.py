import os
import numpy as np

from utils.data import GillickDataMaker

if __name__ == "__main__":
    print("WHAT HOW PLAY VAE: Prepare Data")
    gdm = GillickDataMaker()  

    # # SAMPLES
    # midi = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/drummer1/eval_session/6_hiphop-groove6_87_beat_4-4.mid'
    # notes, timings = gdm.encode(midi)
    # print(notes.shape, timings.shape)
    # midiout = gdm.decode(notes=notes, timings=timings)
    # midiout.dump('samples/whp/remake.mid')
    # midiout = gdm.decode(notes)
    # midiout.dump('samples/whp/timings.mid')
    # midiout = gdm.decode(timings=timings)
    # midiout.dump('samples/whp/notes.mid')

    mididir = '/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove/'
    full_notes = []
    full_timings = []
    for root, dirs, files in os.walk(mididir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                # check file has 4-4 in name
                if '4-4' not in file:
                    continue
                notes, timings = gdm.encode(os.path.join(root, file))
                full_notes.append(notes)
                full_timings.append(timings)

    full_notes = np.vstack(full_notes)
    full_timings = np.vstack(full_timings)

    print(full_notes.shape, full_timings.shape) #(11627, 32, 9) (11627, 32, 18)
    # save as datasets/gillick_notes.npy, datasets/gillick_timings.npy
    np.save('datasets/gillick_notes.npy', full_notes)
    np.save('datasets/gillick_timings.npy', full_timings)
