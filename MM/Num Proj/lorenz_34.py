import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

def xdot(x, y, sigma):
    return -sigma*(x - y)

def ydot(x, y, z, r):
    return r*x - y - x*z

def zdot(x, y, z, b):
    return x*y - b*z

# Parameters
r = 100.5
sigma = 3
b = 1
x0 = 0
y0 = 1
z0 = 0

### Plot 3d trajectory
t_eval = np.linspace(1e-6, 50 ,10000)
t_start = t_eval[0]
t_end = t_eval[-1]

# System of equations
def lorenz(t, state):
    x, y, z = state
    return [xdot(x, y, sigma), ydot(x, y, z, r), zdot(x, y, z, b)]

# Solve the system
initial_conditions = [x0, y0, z0]
solution = solve_ivp(lorenz, [t_start, t_end], initial_conditions, t_eval=t_eval)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2])
# ax.plot(-4, -4, 16, 'ro')  # Marking the point (-4, -4, 16)
ax.set_title('Phase Space Trajectory')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.show()
plt.cla()
plt.clf()
plt.close()

### Lemme do something, plot z vx x trajectory for r = 99.65 and 100.5
r2 = 99.65
sol2 = solve_ivp(lorenz, [t_start, t_end], initial_conditions, t_eval=t_eval)

fig, ax = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'wspace': 0.0})
ax[1].plot(solution.y[0], solution.y[2], label='r = 100.5')
ax[0].plot(sol2.y[0], sol2.y[2], label='r = 99.65')

plt.show()

