import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad


#### Root finding on function so solve a system of equations
def NFW(r, ps, rs):
    """
    NFW profile function
    """
    return ps / ((r / rs) * (1 + r / rs)**2)

def mass(rmax, ps, rs):
    """
    Mass enclosed within radius r for NFW profile
    """
    integrand = lambda r: 4 * np.pi * r**2 * NFW(r, ps, rs)
    return quad(integrand, 0, rmax)[0]

mass_r50 = 5.4e11  ### M_sun out to 50kpc
mass_r230 = 1.9e12  ### M_sun out to 230kpc

def composite(vector):
    ps, rs = vector
    return [mass(50, ps, rs) - mass_r50, mass(230, ps, rs) - mass_r230]

solutions = fsolve(composite, [1e3, 1e3])
print(f"Best fit parameters: ps = {solutions[0]}, rs = {solutions[1]}")
sanity_check = composite(solutions)
print(f"Sanity check (should be close to 0): {sanity_check}")

mass_r50_check = mass(50, solutions[0], solutions[1])
print(f"Mass enclosed within 50 kpc: {mass_r50_check:.2e} M_sun (should be close to {mass_r50:.2e} M_sun)")
