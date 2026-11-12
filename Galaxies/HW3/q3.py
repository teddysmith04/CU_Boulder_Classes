import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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

### Constant from before
K = 7.67e4 ## Msun/pc
M0 = 3.7e6 ## BH mass in Msun
G = 4.302e-3 ## pc Msun^-1 (km/s)^2

def Mr(r):
    """
    Mass enclosed within radius r for Hernquist profile
    """
    return 4 * np.pi * K * r + M0

def vr(r):
    """
    Circular velocity at radius r for Hernquist profile
    """
    return np.sqrt(G * Mr(r) / r)

### Define r range in logspace
rs = np.logspace(-1, 3, 1000) ## pc
mass_enclosed = Mr(rs)

##plot Mr vs r
fig, ax = plt.subplots(2, figsize=(8, 6), gridspec_kw={'hspace': 0})
ax[0].loglog(rs, mass_enclosed, color='blue', lw=2)
ax[0].set_xlabel(r'$r$ [pc]')
ax[0].set_ylabel(r'$M(<r)$ [$M_\odot$]')

ax[0].xaxis.tick_top()
ax[0].xaxis.set_label_position('top')




### Make vs vs r


## equate 4piGK to GM0/r
def vbh(r):
    """
    Circular velocity due to black hole at radius r
    """
    return np.sqrt(G * M0 / r)

vconst = np.sqrt(4 * np.pi * G * K)

vbhs = vbh(rs)
def vroot(r):

    return vbh(r) - vconst

## find roots to find r where vbh = vconst
root = fsolve(vroot, 2)[0]
vs = vr(rs)

ax[1].loglog(rs, vs, color='red', lw=2)
ax[1].axvline(root, color='black', ls='--', label=f'$r_{{eq}}$ = {root:.2f} pc')
ax[1].set_xlabel(r'$r$ [pc]')
ax[1].set_ylabel(r'$v_c$ [km/s]')
ax[1].legend()

plt.savefig('/Users/teddysmith/CU Boulder/1st Year/Galaxies/HWs/hw3/q3.pdf', bbox_inches='tight')
plt.show()


