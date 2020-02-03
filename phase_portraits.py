from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

G = 6.67408 * 10 ** - 11  # gravitational constant
x0 = 127764396  # initial radius of particle
w0 = 115100000 * x0 ** -1.5  # initial angular velocity of particle
M = 5.68319 * 10 ** 26  # mass of saturn
m = 100.0  # mass of particle
n = 37505676206690400000  # mass of mimas
z = 185539000  # distance of mimas from saturn
L = 2.12 * 10 ** 12  # angular momentum of particle
alpha = (G * M / np.pi * (x0 ** 3)) - (np.sqrt(G * M) * w0) / (np.pi * (x0 ** 1.5))  # angular acceleration of particle
w1 = np.sqrt((G * M) / (z ** 3))


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


time_span = np.linspace(0, 10 ** 6, 10 ** 6)  # creating an array of all the times being used

solution = odeint(solver, [x0, np.sqrt(G * M / x0)], time_span, args=(G, x0, w0, m, M, n, z, alpha, L, w1))  # using numerical integration to generate a solution

xs = solution[:, 0]  # extracting the distance part of the solution
vs = solution[:, 1]  # extracting the velocity part of the solution

y1 = np.linspace(-5 * 10 ** 10, 5 * 10 ** 10, 50)  # creates an array for what vectors will be calculated
y2 = np.linspace(-10 ** 7, 10 ** 7, 50)  # creates an array for what vectors will be calculated

Y1, Y2 = np.meshgrid(y1, y2)  # creates an meshgrid the vectors

t = 0  # gives the initial for the phase portrait

u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)  # creates zero arrays with the same size as the vector arrays

NI, NJ = Y1.shape  # gives the dimensions of the Y1 array to new variables

for i in range(NI):  # looping through both dimension to calculate the vectors
    for j in range(NJ):
        # calculating the vector for a given position and time
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = solver([x, y], t, G, x0, w0, m, M, n, z, alpha, L, w1)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

for y20 in np.linspace(0, 10 ** 7, 5):  # looping through predefined values to create curves for
    tspan = np.linspace(0, 10 ** 7, 200)
    y0 = [0.0, y20]
    ys = odeint(solver, y0, time_span, args=(G, x0, w0, m, M, n, z, alpha, L, w1))  # solving the equations using numerical integration
    plt.plot(ys[:, 0], ys[:, 1], 'b-')  # plotting path

#  make the plot
Q = plt.quiver(Y1, Y2, u, v, color='r')
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.gcf().subplots_adjust(left=0.15)
plt.show()
