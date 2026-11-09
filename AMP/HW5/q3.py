import numpy as np
import matplotlib.pyplot as plt
import emcee

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

c_kms = 2.99792458e5
kb, amu = 1.380649e-23, 1.66053906660e-27
K = 2*kb/amu/1e6            ### making the units km^2/s^2


### Define arrays

FWHM = np.array([0.3898, 0.3355, 0.3647, 1.2696, 0.5181, 0.3747,
                 0.4711, 0.9163, 0.4731, 0.6477, 0.4461, 0.5537])  # Å
lam0 = np.array([977, 1032, 1175, 1216, 1335, 1394,
                 1548, 1640, 1808, 1909, 2056, 2175], float)       # Å
A    = np.array([12, 16, 12, 1, 12, 28, 12, 4, 28, 12, 52, 56], float)


#### Calculating the velocity dispersion and preparing the data for MCMC

vD = c_kms * FWHM / lam0 / (2*np.sqrt(2*np.log(2)))
y = vD**2        
x = 1.0 / A         

### Defining the model and log-probability function for MCMC

def model(T, xi, x):
    return K*T*x + xi**2

def log_prob(theta, x, y):
    T, xi, lnf = theta
    if not (1e4 < T < 1e8 and 0 < xi < 500 and -10 < lnf < 2):
        return -np.inf
    mod = model(T, xi, x)
    s2 = (mod*np.exp(lnf))**2  ### fractional scatter (no measured errors)
    return -0.5*np.sum((y-mod)**2/s2 + np.log(s2))

### run MCMC

nwalkers, ndim = 32, 3
p0 = np.array([2e6, 40, -2.0])
pos = p0 + np.array([1e5, 2, 0.1]) * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y))
sampler.run_mcmc(pos, 5000, progress=True)
samples = sampler.get_chain(discard=1000, thin=15, flat=True)

# Posterior summaries
for name, i in [("T [K]", 0), ("xi [km/s]", 1)]:
    lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
    print(f"{name}: {med:.4g} +{hi-med:.3g} -{med-lo:.3g}")

T_lo, T_med, T_hi = np.percentile(samples[:, 0], [16, 50, 84])
xi_lo, xi_med, xi_hi = np.percentile(samples[:, 1], [16, 50, 84])

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
inds = np.random.randint(len(samples), size=100)
for ind in inds:
    sample = samples[ind]
    ax.plot(x, model(sample[0], sample[1], x), color="r", alpha=0.1)
ax.plot(x, model(np.median(samples[:, 0]), np.median(samples[:, 1]), x), color="k", lw=2, label="Median model")
# ax.fill_between(x, model(np.percentile(samples[:, 0], 16), np.percentile(samples[:, 1], 16), x),
                # model(np.percentile(samples[:, 0], 84), np.percentile(samples[:, 1], 84), x), color="k", alpha=0.5, label="Confidence interval (68%)")
ax.plot(x, y, "^", color="b", label="Data")

ax.set(
    xlabel = '1/m (amu)',
    ylabel = r'$c (\Delta \lambda / \lambda_0)^2$',

)
ax.text(0.05, 0.54, r'$T = 2.07^{+0.24}_{-0.23}\times10^6$K', transform=ax.transAxes)
ax.text(0.05, 0.5, r'$\xi =  33.7^{+3.26}_{-3.42}$ km/s', transform=ax.transAxes)
ax.text(0.05, 0.58, r'MCMC median posterior parameter values', transform=ax.transAxes)

ax.legend()
plt.savefig('/Users/teddysmith/CU Boulder/1st Year/AMP/HWs/Hw5/MCMC_fit.pdf', dpi=300, bbox_inches='tight')
plt.show()

