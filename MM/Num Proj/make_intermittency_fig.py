"""Intermittency figure for the Lorenz system (sigma = 10, b = 8/3).

x(t) just above the top of the x^2y^2 window (r_c ~ 166.07) for several r,
and just above the top of the x^2y window (r_c ~ 100.795).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


sigma, b = 10.0, 8.0 / 3.0

def lorenz(t, u, r):
    x, y, z = u
    return [sigma * (y - x), r * x - y - x * z, x * y - b * z]

def xt(r, T=40.0, Ttr=100.0, ic=(0.0, 1.0, 0.0), dt=0.01):
    sol = solve_ivp(lorenz, [0, T + Ttr], ic, args=(r,), rtol=1e-10,
                    atol=1e-12, dense_output=True)
    t = np.arange(Ttr, T + Ttr, dt)
    return t - Ttr, sol.sol(t)[0]

panels = [(166.1, "(a)"), (166.3, "(b)"), (170.0, "(c)"), (100.8, "(d)")]
fig, ax = plt.subplots(len(panels), 1, figsize=(6.5, 8.5), sharex=True, gridspec_kw={"hspace": 0.0})
for a, (r, lab) in zip(ax, panels):
    t, x = xt(r)
    a.plot(t, x, lw=0.7, color="k")
    a.set_ylabel("x(t)")
    a.text(0.02, 0.82, f"{lab} r = {r}", transform=a.transAxes,
           bbox=dict(fc="w", ec="k"))
ax[-1].set_xlabel("t")
plt.tight_layout()
plt.savefig("/Users/teddysmith/CU Boulder/1st Year/MathMeth/Num Proj/fig4.pdf", bbox_inches="tight", dpi=200)

