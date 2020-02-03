from scipy.integrate import odeint
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np

G = 6.67408 * 10 ** - 11  # gravitational constant
x0 = 127764396  # initial radius of particle
k = 9
w0 = 115100000 * x0 ** -1.5  # initial angular velocity of particle
M = 5.68319 * 10 ** 26  # mass of saturn
m = 100.0  # mass of particle
n = 37505676206690400000  # mass of mimas
z = 185539000  # distance of mimas from saturn
L = 2.12 * 10 ** 12  # angular momentum of particle
alpha = (G * M / np.pi * (x0 ** 3)) - (np.sqrt(G * M) * w0) / (np.pi * (x0 ** 1.5))  # angular acceleration of particle
w1 = np.sqrt((G * M) / (z ** 3))

x0s = [
    116882246,
    89197848,
    73631201,
    141592757,
    100726066,
    131988485,
    80486526,
    105467185,
    127764396
]  # array of radii to use


def solver(w, t, G, x0, w0, m, M, n, z, alpha, L, w1):
    x = w[0]  # getting distance for this iteration
    v = w[1]  # getting velocity for this iteration

    h = (w0 - w1) * t + 0.5 * alpha * (t ** 2)
    p1 = 3 * G * M / (x0 ** 2)
    p2 = -4 * (L ** 2) / ((x0 ** 3) * (m ** 2))
    p3 = (G * n * (x0 - z * np.cos(h))) / (((x0 ** 2) + (z ** 2) - 2 * x0 * z * np.cos(h)) ** 1.5)
    p4 = x0 * G * n * (4 * (x0 ** 2) + (z ** 2) - 8 * x0 * z * np.cos(h) + 3 * (z ** 2) * np.cos(2 * h))
    p5 = 2 * ((x0 ** 2) + (z ** 2) - 2 * x0 * z * np.cos(h)) ** 2.5
    p6 = -2 * G * M / (x0 ** 3)
    p7 = 3 * (L ** 2) / ((x0 ** 4) * (m ** 2))
    p8 = -G * n * (2 * (x0 ** 2) + (z ** 2) - 4 * x0 * z * np.cos(h) + 3 * (z ** 2) * ((np.cos(h)) ** 2))
    p9 = ((x0 ** 2) + (z ** 2) - 2 * x0 * z * np.cos(h)) ** 2.5

    acceleration = p1 + p2 + p3 + (p4 / p5) + x * (p6 + p7 + (p8 / p9))  # calculation the equation

    return [v, acceleration]


def fit_sin(tt, yy):
    """
    A function that extracts the equation of the best fitting sine curve to points given
    """

    # creating empty arrays for the times, and values to fill up
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing of peaks
    Fyy = abs(np.fft.fft(yy))  # does a fast fourier transform to extract peaks
    # guess values for the equation, which makes the algorithm faster since it doesn't need to iterate much if it's already near
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  # produces a general sine equation
        return A * np.sin(w * t + p) + c

    # uses optimise curve function from scipy to iteratively get the closest approximation to the equation
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c  # produces the function
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}  # returns all needed values


time_span = np.linspace(0, 10 ** 6, 10 ** 6)  # creating an array of all the times being used

for x in x0s:  # looping through all the values of the initial semi - major axis of the particle to be tested
    solution = odeint(solver, [x, np.sqrt(G * M / x)], time_span, args=(G, x, w0, m, M, n, z, alpha, L, w1))  # using numerical integration to generate a solution

    xs = solution[:, 0]  # extracting the distance part of the solution
    vs = solution[:, 1]  # extracting the velocity part of the solution

    res = fit_sin(time_span, xs)  # getting the equation of the wave
    omega = res['omega']  # getting the angular frequency
    amplitude = res['amp']  # getting the amplitude of the wave
    period = 2 * np.pi / omega  # finding the period from the angular frequency
    print(x, period, amplitude)  # printing the radius used, the period and the amplitude


# make the graph
plt.xlabel('time/s')
plt.ylabel('distance from Saturn/m')
plt.axvline(x=0.0, color=(0, 0, 0))
plt.axhline(y=0.0, color=(0, 0, 0))
plt.title('Distance a ring particle gets from Saturn during orbit')
plt.plot(time_span, xs, label='orbit', color='blue')
plt.legend(loc='best')
plt.show()

