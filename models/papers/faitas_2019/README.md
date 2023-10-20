transformer-xl - the XL model implements a segment-level recurrence mechanism, where the hidden state learnt for each segment is cached and made available to the next segment. Applying this mechanism to every two segments creates a recurrence that effectively spans the length of all segments.

Magenta Groove MIDI Dataset 

a continuous one-dimensional stream of tokens, unique identifiers with a one-to-one mapping to pitch, velocity or time.

all sequences are quantized to 1/16ths 

reduced to 9 unique pitches in total: kick drum, snare drum, closed hi-hat, open hi-hat, low tom, mid tom, high tom, crash

Velocity Bucketing - 4 buckets


Time Tokenisation -  time tokens are inserted into the sequence to separate the pitch-velocity 

All of our sequences are converted to this one-dimensional format and joined together into one long stream. 

seq = [pv1, <t2 −t1 >, pv2, <t3 −t2 >, ..., <tN −tN−1 >, pvN] for n in [1..N ]

