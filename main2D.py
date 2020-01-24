from scipy.integrate import odeint
import scipy as sci
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

G = 6.67408 * 10 ** - 11  # gravitational constant


m1 = 1.989 * 10 ** 30  # sun
m2 = 1.898 * 10 ** 27  # jupiter
m3 = 10 ** 15  # mass of asteroid

n = 2

r20 = 778570000000
r30 = r20 * np.cbrt(1 / n ** 2)

r1 = [0.0, 0.0]
r2 = [r20, 0.0]
r3 = [r30, 0.0]

r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")
r3 = np.array(r3, dtype="float64")

r_com = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)

v1 = [0.0, 0.0]
v2 = [0.0, np.sqrt(G * m1 / r20)]
v3 = [0.0, np.sqrt(G * m1 / r30)]

v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")
v3 = np.array(v3, dtype="float64")

v_com = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)


def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:2]
    r2 = w[2:4]
    r3 = w[4:6]
    v1 = w[6:8]
    v2 = w[8:10]
    v3 = w[10:12]
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

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


init_params = np.array([r1, r2, r3, v1, v2, v3])
init_params = init_params.flatten()
time_span = np.linspace(0, 10 ** 10, 10 ** 6)

three_body_sol = odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3))

r1_sol = three_body_sol[:, :2]
r2_sol = three_body_sol[:, 2:4]
r3_sol = three_body_sol[:, 4:6]

rcom_sol = (m1 * r1_sol + m2 * r2_sol + m3 * r3_sol) / (m1 + m2 + m3)

r1com_sol = r1_sol - rcom_sol
r2com_sol = r2_sol - rcom_sol
r3com_sol = r3_sol - rcom_sol

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(111)

ax.plot(r1com_sol[:, 0], r1com_sol[:, 1], color="darkblue")
ax.plot(r2com_sol[:, 0], r2com_sol[:, 1], color="tab:red")
ax.plot(r3com_sol[:, 0], r3com_sol[:, 1], color="tab:green")

ax.scatter(r1com_sol[-1, 0], r1com_sol[-1, 1], color="darkblue", marker="o", s=100,
           label="Sun")
ax.scatter(r2com_sol[-1, 0], r2com_sol[-1, 1], color="tab:red", marker="o", s=100,
           label="Jupiter")
ax.scatter(r3com_sol[-1, 0], r3com_sol[-1, 1], color="tab:green", marker="o", s=100,
           label="Asteroid")

ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.legend(loc="upper left", fontsize=14)

plt.show()
