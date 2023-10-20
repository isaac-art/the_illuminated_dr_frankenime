import os
import numpy as np
import pretty_midi

class WarrenMIDIMarkov:
    def __init__(self, quantize_to='16th'):
        self.quantize_to = quantize_to

    def midi_to_markov_state(self, mid):
        velocities = [note.velocity for instrument in mid.instruments for note in instrument.notes]
        binary_representations = [format(velocity, '07b') for velocity in velocities]
        reduced_binary_representations = [binary_representations[:4] for binary_representations in binary_representations]
        concatenated_binary = ''.join(reduced_binary_representations)
        markov_state = int(concatenated_binary, 2)
        return markov_state

    def quantize_collapse(self, mid):
        tempo = mid.estimate_tempo()
        beats_per_second = tempo / 60.0
        seconds_per_beat = 1 / beats_per_second
        quantize_multiplier = {'4th': 1, '8th': 0.5, '16th': 0.25}.get(self.quantize_to, 0.25)
        quantize_time = seconds_per_beat * quantize_multiplier

        for instrument in mid.instruments:
            for note in instrument.notes:
                note.start = round(note.start / quantize_time) * quantize_time
                note.end = round(note.end / quantize_time) * quantize_time
        return mid

    def check_file_constraints(self, mid):
        time_sigs = mid.time_signature_changes
        if not any(ts.numerator == 4 and ts.denominator == 4 for ts in time_sigs):
            return False
        if mid.get_end_time() / (4 * mid.estimate_tempo() / 60) < 8:
            return False
        return True

    def encode(self, midi_file):
        try:
            mid = pretty_midi.PrettyMIDI(midi_file)
            if not self.check_file_constraints(mid):
                print("❌")
                return False, None
            print("✅")
            mid = self.quantize_collapse(mid)
            markov_state = self.midi_to_markov_state(mid)
            return True, markov_state
        except:
            print("File Error", midi_file)
            return False, None


if __name__ == "__main__":
    data_root = "/Users/isaac/Library/CloudStorage/Dropbox/nime_ml/gen_dnn_implementations/_datasets/groove"
    warrenMIDIMarkov = WarrenMIDIMarkov(quantize_to='16th')
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".mid"):
                res, encoded = warrenMIDIMarkov.encode(os.path.join(root, file))
                if res: 
                    print(encoded)
                    exit()