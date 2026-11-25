import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

# --- parameters ---
D, k, xc = 0.015, 0.24, 0.35
x0, sigma0 = 0.74, 0.06
T  = 25.0      # years
N  = 2000       # number of cells
dt = 0.05

# --- grid (cell-centered finite volume) ---
dx = 1.0 / N
x      = (np.arange(N) + 0.5) * dx   # cell centers: where u lives
x_face = np.arange(N + 1) * dx       # faces: where flux J lives

# --- initial condition ---
u = np.exp(-(x - x0)**2 / (2 * sigma0**2))
u /= np.sum(u) * dx

def build_L(N, dx, x_face, D, k, xc):
    """
    Return sparse (N x N) matrix L so that du/dt = L @ u.

    Finite-volume idea:  du_i/dt = -(J_{i+1/2} - J_{i-1/2}) / dx
    with the flux at each interior face approximated using u_i and u_{i+1}.
    """
    #for each interior face, write J in terms of u_i and u_{i+1}
    #       (central difference for u_x, average of neighbors for u).

    ## Know J = D*u_x + k(x - xc)*u
    xfi = x_face[1:-1]

    a = D / dx - k * (xfi - xc) / 2  # for right J
    b = -D / dx - k * (xfi - xc) / 2  # for left J

    upper = -b / dx
    main = np.concatenate(([-a[0]], b[:-1] - a[1:], [b[-1]])) / dx
    lower = a / dx

    L = sp.diags([lower, main, upper], [-1, 0, 1], format="csc")

    return L, [upper, main, lower]

L, _ = build_L(N, dx, x_face, D, k, xc)



# --- sanity check before going any further ---
# TODO: what should the column sums of L be if probability is conserved?
print(np.abs(L.sum(axis=0)).max())


# --- Crank-Nicolson matrices ---
I = sp.identity(N, format="csc")
A = I - 0.5 * dt * L
B = I + 0.5 * dt * L
lu = splu(A)

# --- time loop ---
nsteps = int(round(T / dt))
# TODO: allocate storage for whatever you want to track
#       (the mean, the variance, P(playoffs), snapshots of u, ...)
mean_winpct_x = np.zeros(nsteps)
p50 = np.zeros(nsteps)
p40 = np.zeros(nsteps)
p30 = np.zeros(nsteps)
p20 = np.zeros(nsteps)
variance_winpct_x = np.zeros(nsteps)

for n in range(nsteps):
    # one CN step using lu.solve and B
    u = lu.solve(B @ u)
    winpct = np.sum(x*u) * dx
    mean_winpct_x[n] = winpct

    P_playoffs = np.sum(u[x > 0.5]) * dx
    p50[n] = P_playoffs
    p40[n] = np.sum(u[(x > 0.4) & (x < 0.5)]) * dx
    p30[n] = np.sum(u[(x > 0.3) & (x < 0.4)]) * dx
    p20[n] = np.sum(u[x <0.3]) * dx

    win_variance = np.sum((x - winpct)**2 * u) * dx
    variance_winpct_x[n] = win_variance
    
    print(f"t={n*dt:.2f} years: win pct = {winpct:.4f}")
    

### Plot playoff probability over time
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, gridspec_kw={'wspace': 0.0})
ax[0].plot(np.arange(nsteps) * dt + 2002, p50, color='k', lw=2, label = r'P(winpct > 50%)', ls = '-')
ax[0].plot(np.arange(nsteps) * dt + 2002, p40, color='k', lw=2, label = r'P(40 < winpct < 50%)', ls = '--')
ax[0].plot(np.arange(nsteps) * dt + 2002, p30, color='k', lw=2, label = r'P(30 < winpct < 40%)', ls = ':')
ax[0].plot(np.arange(nsteps) * dt + 2002, p20, color='k', lw=2, label = r'P(winpct < 30%)', ls = '-.')
ax[0].set(
    # xlabel='Time (years)',
    ylabel='Probability of Making Playoffs',
    title='Probability of Making Playoffs Over Time',
    xlim=(2002, T + 2002),
    ylim=(0, 1)
)
ax[1].plot(np.arange(nsteps) * dt + 2002, variance_winpct_x, color='r', lw=2, label = r'Mean Win Percentage', ls = '-')
ax[1].set(
    xlabel='Time (years)',
    ylabel='Variance of Win Percentage',
    title='Variance of Win Percentage Over Time',
    xlim=(2002, T + 2002),
)
ax[0].legend()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
fig.tight_layout()
plt.show()




