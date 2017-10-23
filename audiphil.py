# The library with audio processing tools

from collections import defaultdict
import audio_features as fea
import file_processing as iop
import scikits.audiolab

def split_into_channels(sig, num_chan=2):
    outputs = defaultdict(list)

    for i in xrange(0, len(sig), num_chan):
        for j in xrange(num_chan):
            outputs[j].append(sig[i+j])

    return [a(outputs[x]) for x in outputs]

def play_signal(signal, fs = 44100):
    scikits.audiolab.play(signal, fs=fs)

