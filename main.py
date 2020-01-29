from scipy.integrate import odeint
import scipy as sci
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

G = 6.674 * 10 ** - 11  # gravitational constant

x0 = 127764000  # initial r for particle
k = '7over4'
w0 = 195100000 * x0 ** -1.5
M = 5.683 * 10 ** 26
m = 100.0
n = 3.751 * 10 ** 19
z = 185539000
alpha = 1.0
phi = np.pi
a = 1.18 * 10 ** 8
L = 2.12 * 10 ** 12


def solver(w, t, G, a, w0, m, M, n, z, alpha, phi, L):
    x = w[0]
    v = w[1]

    h = w0 * t + 0.5 * alpha * (t ** 2) - phi

    p1 = 3 * G * M / (a ** 2)
    p2 = -4 * (L ** 2) / ((a ** 3) * (m ** 2))
    p3 = (G * n * (a - z * np.cos(h))) / (((a ** 2) + (z ** 2) - 2 * a * z * np.cos(h)) ** 1.5)
    p4 = a * G * n * (4 * (a ** 2) + (z ** 2) - 8 * a * z * np.cos(h) + 3 * (z ** 2) * np.cos(2 * h))
    p5 = 2 * ((a ** 2) + (z ** 2) - 2 * a * z * np.cos(h)) ** 2.5
    p6 = -2 * G * M / (a ** 3)
    p7 = 3 * (L ** 2) / ((a ** 4) * (m ** 2))
    p8 = -G * n * (2 * (a ** 2) + (z ** 2) - 4 * a * z * np.cos(h) + 3 * (z ** 2) * ((np.cos(h)) ** 2))
    p9 = ((a ** 2) + (z ** 2) - 2 * a * z * np.cos(h)) ** 2.5

    acc = p1 + p2 + p3 + (p4 / p5) + x * (p6 + p7 + (p8 / p9))

    return [v, acc]


time_span = np.linspace(0, 10 ** 5, 10 ** 7)

solution = odeint(solver, [x0, 1], time_span, args=(G, a, w0, m, M, n, z, alpha, phi, L))

xs = solution[:, 0]
vs = solution[:, 1]

plt.xlabel('time/s')
plt.ylabel('distance from Saturn/m')
plt.axvline(x=0.0, color=(0, 0, 0))
plt.axhline(y=0.0, color=(0, 0, 0))
plt.title('Distance a ring particle gets from Saturn during orbit')
plt.plot(xs, vs, label='orbit', color='blue')

#  plt.savefig('saturn_k={}.png'.format(k))
plt.show()
