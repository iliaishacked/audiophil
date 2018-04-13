# The utilities used to operate on files

import numpy as np

def load_16bit_pcm(path, lower_endian = True):
    # Can easily abstract it away. Maybe do it later
    endianness = '<' if lower_endian else '>'
    return np.memmap(path, dtype='{}h'.format(endianness), mode='r')

def write_16bit_pcm(data, path, mx = 32767, lower_endian = True):
    outf = open(path, "w")

    endianness = '<' if lower_endian else '>'
    for dt in data:
        outf.write(struct.path('{}h'.format(endianness), int(dt*mx)))

    outf.close()
