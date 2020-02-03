from scipy.integrate import odeint
import scipy as sci
import matplotlib.pyplot as plt
import numpy as np

G = 6.67408 * 10 ** - 11  # gravitational constant
m1 = 1.989 * 10 ** 30  # mass of sun
m2 = 1.898 * 10 ** 27  # mass of jupiter
m3 = 10 ** 26  # mass of asteroid

n = 5
r20 = 778570000000  # initial semi-major axis of jupiter
r30 = r20 * np.cbrt(1 / n ** 2)  # initial semi-major axis of asteroid

# defining initial positions for all three bodies
r1 = [0.0, 0.0]
r2 = [r20, 0.0]
r3 = [r30, 0.0]

# converting arrays to numpy arrays
r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")
r3 = np.array(r3, dtype="float64")

# defining initial velocities for all three bodies
v1 = [0.0, 0.0]
v2 = [0.0, np.sqrt(G * m1 / r20)]
v3 = [0.0, np.sqrt(G * m1 / r30)]

# converting arrays to numpy arrays
v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")
v3 = np.array(v3, dtype="float64")


# defining the function that will calculate the acceleration and velocity of each body after each iteration
def ThreeBodyEquations(w, t, G, m1, m2, m3):
    # extracting the positions and velocities
    r1 = w[:2]
    r2 = w[2:4]
    r3 = w[4:6]
    v1 = w[6:8]
    v2 = w[8:10]
    v3 = w[10:12]

    # setting the position and velocity of the sun to zero, since it is assumed that is stays fixed
    r1 = [0.0, 0.0]
    r1 = np.array(r1, dtype="float64")
    v1 = [0.0, 0.0]
    v1 = np.array(v1, dtype="float64")

    # calculating the distance between each body
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    # calculating the new accelerating and velocity of all three bodies
    dv1bydt = G * m2 * (r2 - r1) / r12 ** 3 + G * m3 * (r3 - r1) / (r13 ** 3)
    dv2bydt = G * m1 * (r1 - r2) / r12 ** 3 + G * m3 * (r3 - r2) / (r23 ** 3)
    dv3bydt = G * m1 * (r1 - r3) / r13 ** 3 + G * m2 * (r2 - r3) / (r23 ** 3)
    dr1bydt = v1
    dr2bydt = v2
    dr3bydt = v3
    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs


# putting the initial conditions into a numpy flattened array
init_params = np.array([r1, r2, r3, v1, v2, v3])
init_params = init_params.flatten()
time_span = np.linspace(0, 3600 * 24 * 365 * 1000, 10 ** 6)  # creating an array of all the times being used

# solving the equations using numerical integration
three_body_sol = odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3))

# extracting the positions of the bodies from the solution
r1_sol = three_body_sol[:, :2]
r2_sol = three_body_sol[:, 2:4]
r3_sol = three_body_sol[:, 4:6]

# calculating the orbital period of jupiter and the asteroid
T2 = int(2 * np.pi * np.sqrt((r20 ** 3) / (G * m1)) * 1000 / (3600 * 24 * 365))
T3 = int(2 * np.pi * np.sqrt((r30 ** 3) / (G * m1)) * 1000 / (3600 * 24 * 365))

# extracting the first orbit of jupiter and the asteroid
firstOrbit2 = r2_sol[:T2]
firstOrbit3 = r3_sol[:T3]

# extracting the last orbit of jupiter and the asteroid
lastOrbit2 = r2_sol[len(r2_sol[0]) - T2:]
lastOrbit3 = r3_sol[len(r3_sol[0]) - T3:]

# plotting the first orbits of jupiter and the asteroid
plt.plot(firstOrbit2[:, 0], firstOrbit2[:, 1], color="tab:red")
plt.plot(firstOrbit3[:, 0], firstOrbit3[:, 1], color="tab:green")

# plotting the last orbits of jupiter and the asteroid
plt.plot(lastOrbit2[:, 0], lastOrbit2[:, 1], color="tab:blue")
plt.plot(lastOrbit3[:, 0], lastOrbit3[:, 1], color="tab:purple")

# plotting the position of the sun
plt.scatter(0, 0)

# showing the plot
plt.show()
