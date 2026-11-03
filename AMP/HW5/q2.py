"""
Script for computing einstien coefficient A_ij by computing matrix element M_ij
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import assoc_laguerre, factorial, assoc_legendre_p

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
Aij_test1 = Aij_calculation(1, 0, 0, 2, 1)


print(f"Test case 1 (2p -> 1s): A_ij = {Aij_test1[0]:.3e}, |M_ij|^2 = {Aij_test1[1]:.3e}, nu_0 = {Aij_test1[2]:.3e} Hz")









    







