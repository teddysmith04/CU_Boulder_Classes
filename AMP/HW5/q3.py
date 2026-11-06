import numpy as np 
import matplotlib.pyplot as plt
import emcee


### Define physical constants
c = 2.99792458e8  # Speed of light in m/s
kb = 1.380649e-23  # Boltzmann constant in J/K

#### Define arrays for the lines and their fwhms
lines = ['CIIIa', 'OVI', 'CIIIb', 'HI', 'CII', 'SiIV', 'CIVa', 'HeII', 'SiII', 'CIIIc', 'CrII', 'FeII']
FWHMs = [0.3893, 0.3355, 0.3647, 1.2696, 0.5181, 0.3747, 0.4711, 0.9163, 0.4731, 0.6477, 0.4461, 0.5537] ## Angstroms
lam0 = [977, 1032, 1175, 1216, 1335, 1393, 1548, 1640, 1808, 1909, 2056, 2175] ## angstroms
masses = [12, 16, 12, 1, 12, 28, 12, 4, 28, 12, 52, 56] ## atomic mass units



### convert FWHMs to m/s using the formula FWHM = (lam0/c) * v, where v is the velocity in m/s
FWHMs = np.array(FWHMs) * 1e-10 * c / (np.array(lam0) * 1e-10) # convert to m/s

## convert lam0s to meters
lam0 = np.array(lam0) * 1e-10  # convert to meters

### convert masses to kg
masses = np.array(masses) * 1.66053906660e-27  # convert to kg

### Perform max likelyfhood estimation of the FWHM for each line using MCMC to find T and microturbulent velocity
def log_likelihood(theta, x, y, yerr):
    m, b, log_scatter = theta
    ### Define model where x = lam_0^2/m, m = T, b = microturbulent velocity^2, log_f = log of the scatter
    model = (2*kb / (c**2)) * m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_scatter)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

### Find the min and max values of lam0**2 /m to set the bounds for the priors
min_x = np.min(lam0**2 /masses)
max_x = np.max(lam0**2 /masses)

x_eval = np.linspace(min_x, max_x, 100)

def log_prior(theta):
    T, xi, log_scatter = theta

    ### Define priors
    if 8e4 < T < 1e9 and 10 < xi < 1e4 and -10.0 < log_scatter < 2.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):  # a function of model parameters and data
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

### Set up the MCMC sampler
pos = [np.array([1e5, 100, -1.0]) + 1e-4 * np.random.randn(3) for i in range(32)]  # Initial positions of walkers
nwalkers, ndim = 32, 3
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(np.array(lam0)**2, np.array(FWHMs), 0.1*np.array(FWHMs)))
sampler.run_mcmc(pos, 5000, progress=True)  # 

### Get the samples and plot the results
samples = sampler.get_chain(discard=1000, thin=15, flat=True)


inds = np.random.randint(len(samples), size=100)

fig, ax = plt.subplots(1, figsize=(10, 8), sharex=True)
for ind in inds:
    sample = samples[ind]
    ax.plot(x_eval, (2*kb / (c**2)) * sample[0] * x_eval + sample[1], color="r", alpha=0.1)

ax.errorbar(lam0**2/masses, FWHMs**2, yerr=0.1*FWHMs, fmt=".k", capsize=0)

plt.show()

