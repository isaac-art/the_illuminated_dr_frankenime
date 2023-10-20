Score-Transformer: A Deep Learning Aid for Music Composition

Jeffrey A. T. Lupker

2021

-----

transformer-decoder architecture
relative positional encoding
250000steps

polyphonic midi

finetuneable

manipulatable: temperature, topk, topp

-----
dataset
"ST was trained from scratch upon a dataset of ~180,000 MIDI recordings of music (using the MAESTRO dataset [5], the Lakh dataset[6], and other free MIDI recordings from imslp.com)"

data processing
"an encoding process modeled after work by Oore et al.[7] provided a method whereby four MIDI musical events were categorized and stored in a dictionary set. This elegant solution consists of 256 note events (note-on and note-off), a single velocity value, and 100 distances in time (10 ms each) between these events referred to as “time-shifts” by Oore et al."

-----
References
TODO
lupker
oore
