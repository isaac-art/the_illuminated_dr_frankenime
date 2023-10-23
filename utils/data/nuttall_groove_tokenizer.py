import os
import random
from miditoolkit import MidiFile, Note
from pretty_midi import PrettyMIDI, Instrument, Note

class NuttallGrooveTokenizer():
    def __init__(self):
        self.quantize = 16 # 16th notes
        self.roland_to_nuttall_pitch_mapping = {
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
        self.time_tokens_ticks = {
            1: 1, # 1 tick
            2: 10, # 10 ticks
            3: 100, # 100 ticks
            4: 1000, # 1000 ticks
            5: 10000, # 10000 ticks
        }
        self.sequence_division_token = 0
        self.ticks_per_qtr = 480
        self.pitch_list = set(self.roland_to_nuttall_pitch_mapping.values())
        # self.pitch_list = list(range(21, 109))
        self.num_pitches = len(self.pitch_list) #should be 9
        self.num_v_buckets = 4 # 4 velocity buckets
        self.num_pitch_vel_tokens = self.num_pitches * self.num_v_buckets # 36
        self.token_pitch_vel_pairs = self.pitch_velocity_tokens(self.pitch_list, self.num_v_buckets)

    def pitch_velocity_tokens(self, pitch_list, num_v_buckets=4):
        token_pitch_vel_pairs = {}
        counter = len(self.time_tokens_ticks) + 1
        for pitch in pitch_list:
            for v in range(1, num_v_buckets + 1):
                token_pitch_vel_pairs[counter] = (pitch, v)
                counter += 1 
        return token_pitch_vel_pairs

    def velocity_to_bucket(self, velocity):
        if velocity < 32: return 1
        elif velocity < 64: return 2 
        elif velocity < 96: return 3
        return 4 
    
    def bucket_to_velocity(self, bucket):
        if bucket == 1: return random.randint(1, 31) #
        elif bucket == 2: return random.randint(32, 63)
        elif bucket == 3: return random.randint(64, 95)
        elif bucket == 4: return random.randint(96, 127)
        return 0 # should never happen?

    def ticks_to_time_tokens(self, ticks):
        # using self.time_tokens_ticks take the input ticks number 
        # and represent it as a sequence of tokens
        # summing up to that number of ticks
        # e.g. 123 -> [3, 2, 2, 1, 1, 1] 345 -> [3,3,3,2,2,2,2,1,1,1,1]
        if ticks == 0: return []
        tokens = []
        for token, tick_value in sorted(self.time_tokens_ticks.items(), key=lambda x: x[1], reverse=True):
            while ticks >= tick_value:
                tokens.append(token)
                ticks -= tick_value
        return tokens
    
    def midi_dir_to_token_stream(self, midi_dir, max_len=-1):
        token_stream = []
        f_count = 0
        for root, dirs, files in os.walk(midi_dir):
            for file in files:
                if file.endswith('mid') or file.endswith('midi'):
                    f_count += 1
                    tokens = self.encode(os.path.join(root, file))
                    token_stream.append(self.sequence_division_token)
                    token_stream += tokens
                    if max_len > 0 and len(token_stream) > max_len:
                        token_stream = token_stream[:max_len]
                        break
        print(f'encoded {f_count} files')
        if max_len > 0: assert len(token_stream) <= max_len
        return token_stream

    def encode(self, f):
        print(f'encoding {f}')
        midi_obj = MidiFile(f)
        # print("midi tempo:", midi_obj.tempo_changes)
        ticks_per_beat = midi_obj.ticks_per_beat # 480
        ticks_per_16th = ticks_per_beat // 4 # 120
        notes = sorted(midi_obj.instruments[0].notes, key=lambda note: note.start)
        sequence = []
        last_start_time = 0
        pv_token_count = 0
        for ni, note in enumerate(notes):
            # print(f'{ni}/{len(notes)}', end='\r')
            # quantize to 16th notes
            note.start = round(note.start / ticks_per_16th) * ticks_per_16th 
            note.end = round(note.end / ticks_per_16th) * ticks_per_16th
            # map pitch to nuttall range(9)
            note.pitch = self.roland_to_nuttall_pitch_mapping.get(note.pitch, note.pitch)
            # bucket velocity to 4
            note.velocity = self.velocity_to_bucket(note.velocity)
            # get token for (pitch, velocity) pair 
            for token, (p, v) in self.token_pitch_vel_pairs.items():
                if p == note.pitch and v == note.velocity:
                    pv_token = token
                    break
            # add time tokens
            # if note.start > last_start_time:
            # skip if first note
            if ni > 0 and note.start > last_start_time:
                silence_ticks = note.start - last_start_time
                time_tokens = self.ticks_to_time_tokens(silence_ticks)
                sequence.extend(time_tokens)
            sequence.append(pv_token)
            last_start_time = note.start # update last end time
        return sequence
        
    def decode(self, tokens, tempo=120):
        midi_out = PrettyMIDI(initial_tempo=tempo, resolution=480)  # Set resolution to match encoder
        # set ticks per beat
        # print(midi_out.resolution)
        instrument = Instrument(program=0, is_drum=True)
        midi_out.instruments.append(instrument)
        current_time_tick = 0
        for ti, token in enumerate(tokens):
            if token in self.token_pitch_vel_pairs:
                pitch, velocity_bucket = self.token_pitch_vel_pairs[token]
                velocity = self.bucket_to_velocity(velocity_bucket)
                start_time = midi_out.tick_to_time(current_time_tick)
                # end_time = midi_out.tick_to_time(current_time_tick + self.ticks_per_qtr // self.quantize)
                end_time = midi_out.tick_to_time(current_time_tick) #zeroduratino hits
                note = Note(start=start_time, end=end_time, pitch=pitch, velocity=velocity)
                instrument.notes.append(note)
                # current_time_tick += self.ticks_per_qtr // self.quantize  # Make sure to update current_time_tick
            elif token in self.time_tokens_ticks:
                current_time_tick += self.time_tokens_ticks[token]
        return midi_out
