import numpy as np
import nolds

def a(d):
    return np.array(d)

def IEMG(data):
    return np.sum(np.square(data.astype(np.float)))

def MAV(data):
    return np.mean(np.absolute(data.astype(np.float)))

def MMAV1(data):
    N = (data.shape[0])
    s = 0.0
    for i in range(N):
        if 0.25*N <= i <= 0.75*N:
            w = 1
        else:
            w = 0.5
        s += w * abs(float(data[i]))
    return s/N

def MMAV2(data):
    N = (data.shape[0])
    s = 0.0
    for i in range(N):
        if 0.25*N <= i <= 0.75*N:
            w = 1
        elif 0.25*N > i:
            w = 4*i/N
        else:
            w = 4*(i-N)/N
        s += w * abs(float(data[i]))
    return s/N

def VAR(data):
    return np.var([x**2 for x in data])

def RMS(data):
    return np.sqrt(np.mean(np.square(data)))

def WL(data):
    data = data.astype(np.float)
    N = float(data.shape[0])
    return sum([ abs(data[i+1]-data[i]) for i in range(int(N)-1)])

def ZC(data):
    mdata = data.copy().astype(np.float) - 0.5
    return (np.diff(np.sign(mdata)) != 0).sum()

def SSC(data):
    mdata = data.copy().astype(np.float) - 0.5
    return sum(1 for i in range(1, len(mdata)-1) if mdata[i-1]*mdata[i]<0 and mdata[i]*mdata[i+1]<0)

def WAMP(data, threashold=0.2):
    mdata = data.copy().astype(np.float) - 0.5
    return sum(1 for i in range(1, len(mdata)) if abs(mdata[i-1]*mdata[i])>= threashold)

def STDDEV(data):
    data = data.astype(np.float)
    N = float(len(data))
    mean = np.mean(data)

    summation = sum([(x - mean)**2 for x in data])
    qq = summation*(1/(N-1))
    return np.sqrt(qq)

def SSI(data):
    return np.sum(np.square(data))

def absval_temp(data):

    N = float(data.shape[0])

    summation = sum([ abs(data[i+1]-data[i]) for i in range(int(N)-1)])

    return (1/(N-1))*summation

def mean_absval_second_diff(data):
    N = float(data.shape[0])

    summation = sum([ abs(data[i+2]-data[i]) for i in range(int(N)-2)])

    return (1/(N-2))*summation

def mean_absval_third_diff(data):
    N = float(data.shape[0])

    summation = sum([ abs(data[i+3]-data[i]) for i in range(int(N)-3)])

    return (1/(N-3))*summation

def mean_absval_fourth_diff(data):
    N = float(data.shape[0])

    summation = sum([ abs(data[i+4]-data[i]) for i in range(int(N)-4)])

    return (1/(N-4))*summation

def getmean(data):
    return np.mean(data.astype(np.float))

def ENT(data):
    return nolds.sampen(data.astype(np.float))

def get_features_from_stream(
        strea,
        fncs = [MAV, MMAV1, MMAV2, SSI, VAR, RMS, WL, ZC, SSC, WAMP, STDDEV]
        ):

    return a([ f(strea) for f in fncs])

def get_ff():
    return ["MAV", "MMAV1", "MMAV2", "SSI", "VAR", "RMS", "WL", "ZC", "SSC", "WAMP", "STDDEV"]
