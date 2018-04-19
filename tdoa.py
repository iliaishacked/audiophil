import numpy as np

def Gxx(s1, s2):
    pad1 = np.zeros(len(s1)).astype("float")
    pad2 = np.zeros(len(s2)).astype("float")

    cs1 = np.hstack([s1,pad1])
    cs2 = np.hstack([pad2,s2])

    f_s1 = np.fft.fft(cs1)
    f_s2 = np.fft.fft(cs2)

    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c

    return f_s

def corr(s1, s2, maxdiff=None):
    f_s=Gxx(s1,s2)

    res = np.fft.ifft(f_s).real

    return np.argmax(res) - len(s2)

def corr_PHAT(s1, s2, maxdiff=None):
    f_s = Gxx(s1, s2)

    kf_s = f_s/abs(f_s)

    res = np.fft.ifft(kf_s).real

    return np.argmax(res) - len(s2)

def corr_weiner(s1, s2, maxdiff=None):

    f_s = Gxx(a(s1), a(s2))
    f_s_1 = Gxx(a(s1), a(s1))
    f_s_2 = Gxx(a(s2), a(s2))

    c12 = (f_s)**2/(f_s_1*f_s_2)
    kf_s = f_s * abs(c12)

    res = (np.fft.ifft(kf_s)).real

    return np.argmax(res) - len(s2)

def asdf(s1, s2, N=64, maxdiff=None):
    res = (-2*np.fft.ifft(np.fft.fft(s1)*np.conj(np.fft.fft(s2))).real + sum(s1**2) + sum(s2**2))/N;
    return np.argmin(res) - len(s2)

def tde_lms(s1, s2, maxdiff=16, _mu=1e-4):

    cfilt = np.zeros(2*maxdiff)

    for i in xrange(maxdiff, len(s1) - maxdiff):
	x1 = a(s2[i-maxdiff:i+maxdiff])

	err = s1[i] - np.dot(cfilt, x1)
	cfilt = cfilt + _mu*err*x1

    return np.argmax(cfilt) - maxdiff

def tde_aed(s1, s2, maxdiff=16, _mu=1e-3):

    h0 = np.full(maxdiff, 0.5).T
    h1 = np.full(maxdiff, 0.5).T

    h0[maxdiff//2] = 1
    h1[maxdiff//2] = 1

    u = a([h1.T, -h0.T]).T

    for i in xrange(maxdiff, len(s1)):
	x0 = a([ s1[i-k] for k in xrange(maxdiff)]).T
	x1 = a([ s2[i-k] for k in xrange(maxdiff)]).T

	xk = a([x0.T, x1.T]).T

	ek = np.dot(u.T,xk)

	t = u - _mu*np.dot(xk, ek)
	u = (t/np.linalg.norm(t))

    h1 = u[:, 0]
    h0 = u[:, 1]

    dif = np.argmax(h1) - np.argmax(h0)

    return dif


