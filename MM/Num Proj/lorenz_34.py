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

def ydot(x, y, z, R):
    return R*x - y - x*z

def zdot(x, y, z, b):
    return x*y - b*z

# Parameters
r = 155
sigma = 10
b = 8/3
x0 = 0
y0 = 1
z0 = 0

### Plot 3d trajectory

t_start = 150.0
t_end = 500.0


# System of equations
def lorenz(t, state, R):
    x, y, z = state
    return [xdot(x, y, sigma), ydot(x, y, z, R), zdot(x, y, z, b)]

def event_func(t, state, R):
    x, y, z = state
    return x  # Event occurs when x crosses zero
event_func.direction = 1  # Only consider crossings from negative to positive

# Solve the system
initial_conditions = [x0, y0, z0]

def run_lorenz(R):
    pre_sol = solve_ivp(lorenz, [0, 100], initial_conditions, args=(R,), 
                        method='RK45', atol=1e-10, rtol=1e-8)
    s0 = pre_sol.y[:, -1]
    sol = solve_ivp(lorenz, [0, t_end-t_start], s0, args=(R,), events=event_func, 
                    method='RK45', atol=1e-10, rtol=1e-8, dense_output=True)
    crossings = sol.y_events[0] 
    t = np.linspace(0, 60, 20000)  # a short stretch for the x-z portrait
    return sol.sol(t), t, crossings


results1 = run_lorenz(100.0)
results2 = run_lorenz(99.7)
results4 = run_lorenz(99.6)
results_noisy = run_lorenz(99.5)



## Plot it
fig, ax = plt.subplots(4, 2, figsize=(12, 24), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
traj, t, crossings = results1
ax[0][0].plot(traj[0], traj[2], label='r = 100.5')
ax[0][1].plot(crossings[:, 1], crossings[:, 2], 'ro', label='Poincaré Section')
ax[0][1].yaxis.set_label_position("right")
ax[0][1].yaxis.tick_right()

traj2, t, crossings2 = results2
ax[1][0].plot(traj2[0], traj2[2], label='r = 100.5')
ax[1][1].plot(crossings2[:, 1], crossings2[:, 2], 'ro', label='Poincaré Section')
ax[1][1].yaxis.set_label_position("right")
ax[1][1].yaxis.tick_right()

traj4, t, crossings4 = results4
ax[2][0].plot(traj4[0], traj4[2], label='r = 100.5')
ax[2][1].plot(crossings4[:, 1], crossings4[:, 2], 'ro', label='Poincaré Section')
ax[2][1].yaxis.set_label_position("right")
ax[2][1].yaxis.tick_right()

traj_n, t, crossings_n = results_noisy
ax[3][0].plot(traj_n[0], traj_n[2], label='r = 100.5')
ax[3][1].plot(crossings_n[:, 1], crossings_n[:, 2], 'ro', label='Poincaré Section')
ax[3][1].yaxis.set_label_position("right")
ax[3][1].yaxis.tick_right()

ax[0][1].text(
    0.3, 0.9, r'r = 100.5 $\sigma$ = 10 b = 8/3', transform=ax[0][1].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square')
)
ax[1][1].text(
    0.3, 0.9, r'r = 99.7 $\sigma$ = 10 b = 8/3', transform=ax[1][1].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square')
)
ax[2][1].text(
    0.3, 0.9, r'r = 99.6 $\sigma$ = 10 b = 8/3', transform=ax[2][1].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square')
)
ax[3][1].text(
    0.3, 0.9, r'r = 99.5 $\sigma$ = 10 b = 8/3', transform=ax[3][1].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square')
)


for i in range(4):
    ax[i][1].set(
        xlim = (26, 30),
        ylim = (95, 97),
    )





plt.savefig('/Users/teddysmith/CU Boulder/1st Year/MathMeth/Num Proj/per_double.pdf', bbox_inches='tight')
plt.show()
exit()



# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(solution.y[0], solution.y[1], solution.y[2])
# # ax.plot(-4, -4, 16, 'ro')  # Marking the point (-4, -4, 16)
# ax.set_title('Phase Space Trajectory')
# ax.set_xlabel('x(t)')
# ax.set_ylabel('y(t)')
# ax.set_zlabel('z(t)')
# plt.show()
# plt.cla()
# plt.clf()
# plt.close()

### Lemme do something, plot z vx x trajectory for r = 99.65 and 100.5
r2 = 160
def lorenz2(t, state):
    x, y, z = state
    return [xdot(x, y, sigma), ydot(x, y, z, r2), zdot(x, y, z, b)]
sol2 = solve_ivp(lorenz2, [t_start, t_end], initial_conditions, t_eval=t_eval)

fig, ax = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'wspace': 0.0})
ax[1].plot(solution.y[0][2000:], solution.y[2][2000:], label='r = 100.5')
ax[1].text(0.5, 0.9, f'r = {r}', transform=ax[1].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square'))
ax[0].plot(sol2.y[0][2000:], sol2.y[2][2000:], label='r = 99.65')
ax[0].text(0.5, 0.9, f'r = {r2}', transform=ax[0].transAxes, bbox=dict(edgecolor='black', facecolor='white', boxstyle='square'))

plt.show()

### Make a poincare diagram by taking a slice through a plane at z = 100
poincare_z = 100

index_at_poincare = np.where(np.isclose(solution.y[2][5000:], poincare_z, atol=0.1))[0]
poincare_x = solution.y[0][index_at_poincare]
poincare_y = solution.y[1][index_at_poincare]

index_at_poincare2 = np.where(np.isclose(sol2.y[2][5000:], poincare_z, atol=0.1))[0]
poincare_x2 = sol2.y[0][index_at_poincare2]
poincare_y2 = sol2.y[1][index_at_poincare2]



fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(poincare_x, poincare_y, s=50, alpha=1, label = f'r = {r}', color='blue')
ax.scatter(poincare_x2, poincare_y2, s=50, alpha=1, color='orange', label = f'r = {r2}')
ax.set_title(f'Poincaré Section at z = {poincare_z}')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
plt.legend()

plt.show()

