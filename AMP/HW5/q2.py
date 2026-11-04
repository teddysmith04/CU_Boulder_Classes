"""
Script for computing einstien coefficient A_ij by computing matrix element M_ij
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import assoc_laguerre, factorial, assoc_legendre_p
import pandas as pd
from scipy.optimize import curve_fit

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

### Defining constants
a0 = 5.29177210903e-11  # Bohr radius in meters
ep_0 = 8.854187817e-12  # Vacuum permittivity in F/m
h = 6.62607015e-34  # Planck's constant in J*s
c = 2.99792458e8  # Speed of light in m/s
e = 1.602176634e-19  # Elementary charge in C

### Defining normalization functions

def norm_R(n, l):
    """
    Normalization constant for radial wavefunction R(n, l)
    """
    return np.sqrt((2 / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))

def norm_T(l, m):
    """
    Norm for the Theta function Theta_lm
    """
    return np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))


#### Define wave functions
def R(n, l, r):
    """
    Radial wavefunction R(n, l, r)
    """
    norm = norm_R(n, l)
    return norm * np.exp(-r / (n * a0)) * (2 * r / (n * a0))**l * assoc_laguerre(2*r/(n*a0), n-l-1, 2*l+1)

def Theta(l, m, theta):
    """
    Angular wavefunction Theta(l, m, theta)
    """
    norm = norm_T(l, m)
    return norm * assoc_legendre_p(l, m, np.cos(theta))[0]

def A21(Mij_sq, nu_0):
    return (16 * np.pi**3 * nu_0**3 * e**2 * a0**2 / (3 * ep_0 * h * c**3)) * Mij_sq

def Aij_calculation(ni, li, mi, nj, lj): #### mi = mj which also sets the phi integral to just be 2 pi
    """
    Calculate the Einstein coefficient A_ij for a transition from state (ni, li, mi) to (nj, lj, mj)
    args:
        ni, li, mi: quantum numbers for initial state
        nj, lj: quantum numbers for final state (mj = mi)
    """
    ### Calculate r integral = int (R_nj_lj^* r^3 R_ni_li dr)
    max_r = np.max([ni, nj])**2 * a0 ## Max upper bound for integration

    integrand_r = lambda r: R(nj, lj, r) * r**3 * R(ni, li, r)
    R_integral, _ = quad(integrand_r, 0, max_r)

    ### Calculate theta integral = int (Theta_lj_mj^* sin(theta) cos(theta) Theta_li_mi dtheta)
    integrand_theta = lambda theta: Theta(lj, mi, theta) * np.sin(theta) * np.cos(theta) * Theta(li, mi, theta)
    theta_integral, _ = quad(integrand_theta, 0, np.pi)

    ### Know: |M_ij|^2 = M_z^2, and M_z = 2pi * R_integral * theta_integral
    ### Calculate M_ij
    M_z = 2 * np.pi * R_integral * theta_integral
    M_ij_sq = M_z**2 / a0**2  # Convert to dimensionless

    ### Calculate nu_0 = (E_i - E_j)/h
    E_i = -13.6 * e / ni**2
    E_j = -13.6 * e / nj**2

    nu_0 = -(E_i - E_j) / h
    # nu_0 = 1e20

    ### Calculate A_ij
    A_ij = A21(M_ij_sq, nu_0)

    return A_ij, M_ij_sq, nu_0

#### Test cases that should be 0 (1s -> 1s , 3d -> 3d)
Aij_test1 = Aij_calculation(1, 0, 0, 3, 2)

# print(f'Test case Aij: {Aij_test1[0]:.3e}, Mij_sq: {Aij_test1[1]:.3e}, nu_0: {Aij_test1[2]:.3e}')


####_____________________________________________ part c _______________________________________________ ####
### Run Aij calc for grid of nj p -> ni s
### nj = 2, 3, 4, ..., 10, ni = 1, 2, 3, ..., nj-1
### Save to a df table with structure: nj, ni, Aij
df_dict = {
    'nj': np.arange(2, 11),
    'ni': np.arange(1, 10),
    'Aij': np.zeros((9, 9)),
    'Mij_sq': np.zeros((9, 9)),
    'nu_0': np.zeros((9, 9))
}


for ni in df_dict['ni']:
    for nj in df_dict['nj']:
        if ni < nj:
            Aij, Mij_sq, nu_0 = Aij_calculation(ni, 0, 0, nj, 1)
            df_dict['Aij'][nj-2, ni-1] = Aij
            df_dict['Mij_sq'][nj-2, ni-1] = Mij_sq
            df_dict['nu_0'][nj-2, ni-1] = nu_0


df = pd.DataFrame(df_dict['Aij'], index=df_dict['nj'], columns=df_dict['ni'])
### Print or save as a latex table
# print(df.to_latex(index=True, header=True, float_format="%.1e"))





########_____________________________________________  pard d _____________________________________________ ##########
### Fit the data to A_ij = C/(n_i^alpha * n_j^beta), find C, alpha, beta

### Step 1: fix nj, fit Aij vs ni to find alpha and C
nj_fixed = 40
Aij_vals = []
ni_vals = []
for ni in np.arange(1, nj_fixed):
    if ni < nj_fixed:
        Aij, Mij_sq, nu_0 = Aij_calculation(ni, 0, 0, nj_fixed, 1)
        Aij_vals.append(Aij)
        ni_vals.append(ni)


### Linear power law model to fit
def log_power_law(x, C, alpha):
    return alpha * x + C

log_Aij_vals = np.log10(Aij_vals)
log_ni_vals = np.log10(ni_vals)

popt, pcov = curve_fit(log_power_law, log_ni_vals, log_Aij_vals)

###Plot it too 
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.0})
ax[0].plot(log_ni_vals, log_Aij_vals, 'o', label='Data')
ax[0].plot(log_ni_vals, log_power_law(log_ni_vals, *popt), 'r-', label='Fit: C=%.1e, alpha=%.1f' % (10**popt[1], popt[0]))
ax[0].set(
xlabel='log10(ni)', ylabel='log10(Aij)', title=f'Fit for fixed nj={nj_fixed}'
)


# print(f"Fit results for fixed nj={nj_fixed}: C={10**popt[1]:.3e}, alpha={popt[0]:.3f}")


#### Do the opposite for fixed ni, fit Aij vs nj to find beta and C
ni_fixed = 1
nj_max = 40
Aij_vals = []
nj_vals = []
for nj in np.arange(ni_fixed + 1, nj_max + 1):
    if ni_fixed < nj:
        Aij, Mij_sq, nu_0 = Aij_calculation(ni_fixed, 0, 0, nj, 1)
        Aij_vals.append(Aij)
        nj_vals.append(nj)

### Linear power law model to fit
log_Aij_vals = np.log10(Aij_vals)
log_nj_vals = np.log10(nj_vals)

popt, pcov = curve_fit(log_power_law, log_nj_vals, log_Aij_vals)
beta = popt[0]
C = 10**popt[1]

ax[1].plot(log_nj_vals, log_Aij_vals, 'o', label='Data')
ax[1].plot(log_nj_vals, log_power_law(log_nj_vals, *popt), 'r-', label='Fit: C=%.1e, beta=%.1f' % (C, beta))
ax[1].set(
xlabel='log10(nj)', ylabel='log10(Aij)', title=f'Fit for fixed ni={ni_fixed}'
)

ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()

ax[0].legend()
ax[1].legend()
plt.savefig('/Users/teddysmith/CU Boulder/1st Year/AMP/HWs/Hw5/Aij_maxn20.pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"Fit results for fixed ni={ni_fixed}: C={C:.3e}, beta={beta:.3f}")














    







