import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

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
plt.figure(figsize=(12, 8))
plt.plot(solution.t, solution.y[0], label='x(t)')
plt.plot(solution.t, solution.y[1], label='y(t)')
plt.plot(solution.t, solution.y[2], label='z(t)')
plt.title('Lorenz System')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()

# Plotting the phase space trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2])
ax.plot(-4, -4, 16, 'ro')  # Marking the point (-4, -4, 16)
ax.set_title('Phase Space Trajectory')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.show()


### Repeat for r = 25
r = 26
sol2 = solve_ivp(lorenz, [t_start, t_end], initial_conditions, t_eval=t_eval)

plt.figure(figsize=(12, 8))
plt.plot(sol2.t, sol2.y[0], label='x(t)')
plt.plot(sol2.t, sol2.y[1], label='y(t)')
plt.plot(sol2.t, sol2.y[2], label='z(t)')
plt.title('Lorenz System')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()


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
