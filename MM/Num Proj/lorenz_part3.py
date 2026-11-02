"""
Parameter Set 3: the x^2 y period-doubling window of the Lorenz system
(sigma=10, b=8/3, 99.524 < r < 100.795; Sparrow Sec. 4.2).

Poincare section: plane x = 0 crossed with dx/dt > 0 (Sparrow's choice).
One orbit period -> one crossing, so the number of distinct points in the
(y,z) section equals the period multiplicity (1, 2, 4, ...).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True})

sigma, b = 10.0, 8/3
IC = [0.0, 1.0, 0.0]
T_TRANS, T_END = 150.0, 500.0       # discard transient, then record

def rhs(t, s, r):
    x, y, z = s
    return [sigma*(y - x), r*x - y - x*z, x*y - b*z]

def section(t, s, r):                # event function: x = 0
    return s[0]
section.direction = 1                # only x going - to +

def run(r, t_end=T_END, ic=IC):
    # transient first, so the plotted orbit is on the attractor
    pre = solve_ivp(rhs, [0, T_TRANS], ic, args=(r,), method='DOP853',
                    rtol=1e-9, atol=1e-11)
    s0 = pre.y[:, -1]
    sol = solve_ivp(rhs, [0, t_end - T_TRANS], s0, args=(r,), method='DOP853',
                    rtol=1e-9, atol=1e-11, events=section, dense_output=True)
    ev = sol.y_events[0]             # states exactly on x=0
    t = np.linspace(0, 60, 20000)    # a short stretch for the x-z portrait
    return sol.sol(t), t, ev

# (r, label)
cases = [(100.5,  r'stable $x^2y$'),
         (99.65,  r'period 2: $y^2xy^2x$'),
         (99.55,  r'period 4'),
         (99.50,  r'just below $r_\infty$: noisy periodicity'),
         (100.8,  r'just above window: intermittent chaos')]

results = {r: run(r) for r, _ in cases}

# ---- Figure 1: x-z portraits ----
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for ax, (r, lab) in zip(axs.flat, cases):
    traj, t, ev = results[r]
    ax.plot(traj[0], traj[2], lw=0.5)
    ax.set_title(f'r = {r}  ({lab})', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('z')
axs.flat[-1].axis('off')
fig.tight_layout(); fig.savefig('fig1_xz_portraits.png', dpi=200)

# ---- Figure 2: Poincare sections (y,z) at x=0 ----
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for ax, (r, lab) in zip(axs.flat, cases):
    ev = results[r][2]
    ax.scatter(ev[:, 1], ev[:, 2], s=4, alpha=0.6)
    ax.set_title(f'r = {r}  ({len(ev)} crossings)', fontsize=10)
    ax.set_xlabel('y'); ax.set_ylabel('z')
axs.flat[-1].axis('off')
fig.tight_layout(); fig.savefig('fig2_poincare.png', dpi=200)

# ---- Figure 3: time series for intermittency (r=100.8) vs stable ----
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ax, r in zip(axs, [100.5, 100.8]):
    sol, t, ev = run(r, t_end=T_TRANS + 60)[0:3]
    ax.plot(t, sol[0], lw=0.7); ax.set_ylabel('x(t)'); ax.set_title(f'r = {r}')
axs[1].set_xlabel('t'); fig.tight_layout(); fig.savefig('fig3_timeseries.png', dpi=200)

# Figure 4 is slow (~10+ min); set True to generate it
MAKE_BIFURCATION = False
if MAKE_BIFURCATION:
    # ---- Figure 4: bifurcation diagram (y at x=0 crossings vs r) ----
    rs = np.linspace(99.40, 100.90, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in rs:
        pre = solve_ivp(rhs, [0, 120], IC, args=(r,), method='DOP853', rtol=1e-9, atol=1e-11)
        sol = solve_ivp(rhs, [0, 80], pre.y[:, -1], args=(r,), method='DOP853',
                        rtol=1e-9, atol=1e-11, events=section)
        ev = sol.y_events[0][-25:]
        ax.plot(np.full(len(ev), r), ev[:, 2], 'k.', ms=1)
    ax.set_xlabel('r'); ax.set_ylabel('z at x=0 crossing')
    fig.tight_layout(); fig.savefig('fig4_bifurcation.png', dpi=200)


# number of distinct section points (quick period check)
for r, lab in cases:
    ev = results[r][2][-200:]
    n = len(np.unique(np.round(ev[:, 2], 3)))
    print(f'r={r}: distinct z values in last 200 crossings = {n}')
