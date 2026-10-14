import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import expm1, gamma, zeta


#### Useful constants
h = 6.62607015e-34  # Planck's constant in J*s
kB = 1.380649e-23  # Boltzmann's constant in J/K
c = 3e8  # Speed of light in m/s


### Question Constants
Rin = 1e11  # Inner radius in meters
Rout = 1e16  # Outer radius in meters
Tin = 1e6  # Inner temperature in Kelvin
Rmid = np.sqrt(Rin * Rout)  # Geometric mean of Rin and Rout
q = 3/4  # Power-law index for temperature profile

def T(r):
    return Tin * ( Rin / r )**(3/4) * (1 + (r / Rmid)**2)**(1/8)

def Bnu(r, nu):
    x = h * nu / (kB * T(r))
    # denom = 1 / (np.exp(x) - 1) if x < 700 else 0  # Avoid overflow in exp for high frequencies
    return (2 * h * nu**3 / c**2) * np.exp(-x) / (-expm1(-x))

v_range = 3*np.logspace(10, 18, 10000) ### From 0.1 nm to 10^7 nm, but in frequency space
same_but_lam = c *1e9 / v_range ## in nm

def ln_integrand(u, nu):
    r = np.exp(u)
    x = h * nu / (kB * T(r))

    ln_bose = 0 - np.log(expm1(x))
    return (np.log(2*np.pi) + 2*u + np.log(2*h*nu**3/c**2) + ln_bose)

def Inu(nu_0):
    func = lambda u: ln_integrand(u, nu_0)
    lnR = np.linspace(np.log(Rin), np.log(Rout), 2000)
    lmax = np.max(func(lnR))             
    descaled_func = lambda u: np.exp(func(u) - lmax)
    # peak = lnR[np.argmax(func(lnR))]
    val = quad(descaled_func, np.log(Rin), np.log(Rout),
               limit=500)[0]
    return val * np.exp(lmax)

B_rin = Bnu(Rin, v_range)
B_rout = Bnu(Rout, v_range)

I_rin = 2 * np.pi * Rin**2 * B_rin
I_rout = 2 * np.pi * Rout**2 * B_rout

Inu_values = np.array([Inu(nu) for nu in v_range])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(same_but_lam, I_rin, lw=1.5, ls='-.', label=r'$I_\nu(R_{in})$', color = 'b')
ax.plot(same_but_lam, I_rout, lw=1.5, ls='-.', label=r'$I_\nu(R_{out})$', color = 'r')
ax.plot(same_but_lam, Inu_values, lw=2, color = 'k', label=r'$I_\nu$ (integrated)')

ax.set(
    xscale='log',
    yscale='log',
    xlabel=r'Wavelength (nm)',
    ylabel=r'$I_\nu$ (W m$^{-2} \lambda^{-1}$)',
    ylim = (1e15, 1e24)
)
ax.legend()
### Add xaxis on top with wavelengths in nm
# ax2 = ax.twiny()
# ax2.set_xscale('log')
# ax2.set_xlim(ax.get_xlim())
# ax2.set_xticks([c / (1e-9), c / (1e-8), c / (1e-7), c / (1e-6), c / (1e-5), c / (1e-4), c / (1e-3), c / (1e-2)])
# ax2.set_xticklabels(['1 nm', '10 nm', '100 nm', '1 μm', '10 μm', '100 μm', '1 mm', '1 cm'])
# ax2.set_xlabel('Wavelength (nm, μm, mm, cm)')
# ax2.invert_xaxis()  # Invert x-axis to have shorter wavelengths on the right
# ax.invert_xaxis()  # Invert x-axis to have shorter wavelengths on the right


# plt.savefig('HW4/q4.pdf', bbox_inches='tight')
plt.show()



#### Try to do only in log space to avoid err
