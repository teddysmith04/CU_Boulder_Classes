import numpy as np
import matplotlib.pyplot as plt
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

T = 6000 # Temperature in Kelvin

h = 6.62607015e-34  # Planck's constant in J*s
c = 3e8  # Speed of light in m/s

def Bv(v, T):
    """Calculate the spectral radiance of a blackbody at temperature T for frequency v."""
    return (2 * h * v**3 / c**2) / (np.exp(h * v / (1.380649e-23 * T)) - 1)

def Blambda(lmbda, T):
    """Calculate the spectral radiance of a blackbody at temperature T for wavelength lmbda."""
    return (2 * h * c**2 / lmbda**5) / (np.exp(h * c / (lmbda * 1.380649e-23 * T)) - 1)

## lambda and v space to iterate over
lambda_space = np.logspace(-8, -2, 1000)  # Wavelengths from 10 nm 10^7 nm
v_space = c / lambda_space  # Corresponding frequencies

T_range = np.arange(5990, 6010, 1)  # Temperature range from 5990 K to 6010 K

BvT = np.array([Bv(v_space, T) for T in T_range])
BlamT = np.array([Blambda(lambda_space, T) for T in T_range])

### Iterate over each frequency, compute dlogB/dlogT and store results to plot power vs v, lam
a = []
b = []
for i, v in enumerate(v_space):
    dlogB_dlogT_v = np.gradient(np.log(BvT[:, i]), np.log(T_range))
    dlogB_dlogT_lam = np.gradient(np.log(BlamT[:, i]), np.log(T_range))
    a.append(dlogB_dlogT_v)
    b.append(dlogB_dlogT_lam)

median_BvT = np.median(BvT, axis=0)
medain_BlamT = np.median(BlamT, axis=0)

max_v_i = np.argmax(median_BvT)
max_lam_i = np.argmax(medain_BlamT)



max_v = v_space[max_v_i]
max_lam = lambda_space[max_lam_i]

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(lambda_space * 1e9, np.median(b, axis = 1), lw=6, label=r'$\frac{d \log B_\lambda}{d \log T}$', color = 'r', alpha = 0.5)
ax.plot(c / v_space * 1e9, np.median(a, axis = 1), lw=2, label=r'$\frac{d \log B_\nu}{d \log T}$', color = 'k', ls = '--')
ax.set(
    xscale='log',
    xlabel=r'Wavelength (nm)',
    ylabel=r'$\frac{d \log B}{d \log T}$',
)
ax.legend()

a_lim = a[-1]
b_lim = b[-1]

a_median = np.median(a, axis = 1)
b_median = np.median(b, axis = 1)

### find the index of a_median, b_median closest to 4
bol_ind_v = np.argmin(np.abs(a_median - 4))
bol_ind_lam = np.argmin(np.abs(b_median - 4))




bol_v = v_space[bol_ind_v]
bol_lam = lambda_space[bol_ind_lam]

print(f'The Bolometric result of a = b= 4 occurs at {bol_v:.2e} Hz, corresponding to {c / bol_v * 1e9:.2f} nm')
print(f'The Bolometric result of a = b= 4 occurs at {bol_lam * 1e9:.2f} nm, corresponding to {c / bol_lam:.2e} Hz')


print(f'At long wavelengths, B_lambda prop T^{np.median(b_lim):.0f} and B_nu \propto T^{np.median(a_lim):.0f}')
print()
print(f'Maximum of B_nu occurs at {max_v:.2e} Hz, corresponding to {c / max_v * 1e9:.2f} nm')
print(f'Maximum of B_\lambda occurs at {max_lam * 1e9:.2f} nm, corresponding to {c / max_lam:.2e} Hz')

print(b[int(max_lam_i)][:])

plt.savefig('q3.pdf', dpi=300, bbox_inches='tight')
plt.show()




    

