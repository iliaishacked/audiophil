# The library with audio processing tools

from collections import defaultdict
import audio_features as fea
import tdoa
import numpy as np
import itertools
import file_processing as iop
import ml_au as ml
import scikits.audiolab
import matplotlib.pyplot as plt

def split_into_channels(sig, num_chan=2):
    outputs = defaultdict(list)

    for i in xrange(0, len(sig), num_chan):
        for j in xrange(num_chan):
            outputs[j].append(sig[i+j])

    return [np.array(outputs[x]) for x in outputs]

def play_signal(signal, fs = 44100, max_vol=None):
    if max_vol is not None:
        signal = np.array(signal).copy()
        signal = signal/max(np.abs(signal))
        signal = signal*max_vol
    scikits.audiolab.play(signal, fs=fs)


def plot_signal(datas, show=False):

    if type(datas) is tuple:
        datas = [datas]
    fig, ax = plt.subplots(1, facecolor='w', edgecolor='k')

    for i in xrange(len(datas)):
        data, lbl = datas[i]

        ax.plot(data, label = lbl)

    ax.grid()

    if show:
        plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, cm_size=15):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fig, ax = plt.subplots(1, figsize=(cm_size, cm_size), facecolor='w', edgecolor='k')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "%0.2f"% cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "%d"% int(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




