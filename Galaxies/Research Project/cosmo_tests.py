import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology
from colossus.lss import mass_function

# 1. Set the exact MultiDark Planck (MDPL2) cosmology
mdpl2_params = {
    'flat': True,
    'H0': 70.0,
    'Om0': 0.307115,
    'Ob0': 0.048206,
    'sigma8': 0.8228,
    'ns': 0.96
}
cosmo = cosmology.setCosmology('MDPL2', mdpl2_params)
h = cosmo.h
print(h)

# 2. Define halo masses in M_sun/h and redshift
M = np.logspace(11, 15, 50)
z = 0.0

# 3. Compute dn/dln(M) in comoving (Mpc/h)^-3
# 'comparat17' is calibrated directly on MultiDark Planck simulations
dndlnM = mass_function.massFunction(
    M, z, 
    mdef='vir', 
    model='comparat17', 
    q_out='dndlnM'
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(M * h, dndlnM * h**3, label='MultiDark Planck (MDPL2)', color='blue')
ax.set(
    xscale='log',
    yscale='log',
    xlabel='Halo Mass [M_sun/h]',
    ylabel='dn/dln(M) [(Mpc/h)^-3]',
    title='Halo Mass Function at z=0 (MDPL2 Cosmology)'
)
plt.show()

