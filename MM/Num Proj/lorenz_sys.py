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
r = 17
sigma = 3
b = 1
x0 = 0
y0 = 1
z0 = 0

# Time settings
t_eval = np.linspace(1e-6, 50 ,1000)
t_start = t_eval[0]
t_end = t_eval[-1]

# System of equations
def lorenz(t, state):
    x, y, z = state
    return [xdot(x, y, sigma), ydot(x, y, z, r), zdot(x, y, z, b)]

# Solve the system
initial_conditions = [x0, y0, z0]
solution = solve_ivp(lorenz, [t_start, t_end], initial_conditions, t_eval=t_eval)

# Plotting x(t), y(t), z(t)
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.0})
ax1[0].plot(solution.t, solution.y[0], label='x(t)')
ax1[0].plot(solution.t, solution.y[1], label='y(t)')
ax1[0].plot(solution.t, solution.y[2], label='z(t)')
ax1[0].set(
    xlabel='Time',
    ylabel='Values',
)
ax1[0].text(0.5, 0.9, 'r = 17', transform=ax1[0].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square'))
ax1[0].legend()

# plt.figure(figsize=(12, 8))
# plt.plot(solution.t, solution.y[0], label='x(t)')
# plt.plot(solution.t, solution.y[1], label='y(t)')
# plt.plot(solution.t, solution.y[2], label='z(t)')
# plt.title('Lorenz System')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.legend()
# plt.grid()
# plt.show()

# Plotting the phase space trajectory



### Repeat for r = 25
r = 26
sol2 = solve_ivp(lorenz, [t_start, t_end], initial_conditions, t_eval=t_eval)


ax1[1].plot(sol2.t, sol2.y[0], label='x(t)')
ax1[1].plot(sol2.t, sol2.y[1], label='y(t)')
ax1[1].plot(sol2.t, sol2.y[2], label='z(t)')
ax1[1].set(
    xlabel='Time',
    # ylabel='Values',
)
ax1[1].text(0.5, 0.9, 'r = 26', transform=ax1[1].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square'))
ax1[1].yaxis.tick_right()


plt.savefig('/Users/teddysmith/CU Boulder/1st Year/MathMeth/Num Proj/fig1.pdf', bbox_inches='tight')
plt.show()
exit()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2])
ax.plot(-4, -4, 16, 'ro')  # Marking the point (-4, -4, 16)
ax.set_title('Phase Space Trajectory')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.show()
plt.cla()
plt.clf()
plt.close()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol2.y[0], sol2.y[1], sol2.y[2])
ax.plot(-5, -5, 25, 'ro')
ax.plot(5, 5, 25, 'ro') 
ax.set_title('Phase Space Trajectory')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.show()
