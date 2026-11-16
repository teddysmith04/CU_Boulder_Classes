import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fftn, fftshift, fftfreq
from scipy.signal.windows import hann
from matplotlib.ticker import MaxNLocator

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

### ___________________________________________________________________________________________
### Do the easy part first. Pick a pixel, say k_x = k_y = k_a = 0. Plot the power of that pixel as a funciton of temporal frequency w. 
### ___________________________________________________________________________________________

DT = 45.0  # time step [s]

filepath = '/Users/teddysmith/CU Boulder/1st Year/MathMeth/AMP HWs/Codes/HW5 Codes/hmi.rdVtrack_fd15.2121.225.225.0.-22.5.-00.0.V.fits'
file = fits.open(filepath)

data = np.array(file[0].data, dtype=np.float32)

nt, nx, ny = data.shape

data -= data.mean()  # Subtract the mean to remove DC component
data *= hann(nt).astype(np.float32)[:, None, None]
data *= hann(nx).astype(np.float32)[None, :, None] ### window in cyl radius maybe
data *= hann(ny).astype(np.float32)[None, None, :]

## compute 3D fft
fft_data = fftn(data, workers=-1)
del data
power = np.abs(fft_data) ** 2
del fft_data
power = fftshift(power)
 
## Frequency axes, shifted to match the power spectrum
w = fftshift(fftfreq(nt, d=DT)) * 1e3     # temporal frequency [mHz]
ky = fftshift(fftfreq(ny))                # [cycles / pixel]
kx = fftshift(fftfreq(nx))                # [cycles / pixel]

it0, iy0, ix0 = nt // 2, ny // 2, nx // 2   # index of zero frequency on each axis

### loop over the temporal frequency axis and plot the power of the pixel at k_x = k_y = 0 as a function of w.
power_0 = np.zeros(len(w)) ## will add values of power at k_x = k_y = 0 for each temporal frequency
for i in range(len(w)):
    power_0[i] = power[i, iy0, ix0]

### Define range for total wavenumber and frequency
kx_max, kx_min, ky_max, ky_min = kx.max(), kx.min(), ky.max(), ky.min()
ka_min, ka_max = np.sqrt(kx_min**2 + ky_min**2), np.sqrt(kx_max**2 + ky_max**2)
w_min, w_max = w.min(), w.max()

### Define total wavenumber bins
ka_bins = np.linspace(ka_min, ka_max, 50)  # Makes 50 bins - empty bins
w_bins = np.linspace(w_min, w_max, 50)  # Makes 50 bins

power_w_ka_bins = np.zeros((len(ka_bins), len(w_bins)))  # shape (number of ka bins, number of frequencies)

### Vectorized approach to calculate ka and bin indices
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
ka_grid = np.sqrt(kx_grid**2 + ky_grid**2)

for i, wi in enumerate(w):
    w_bin_index = np.digitize(wi, w_bins) -1
    ka_bin_indices = np.digitize(ka_grid.flatten(), ka_bins) - 1
    valid_indices = (0 <= ka_bin_indices) & (ka_bin_indices < len(ka_bins))
    
    # Accumulate power in bins
    power_w_ka_bins[ka_bin_indices[valid_indices], w_bin_index] += power[i].flatten()[valid_indices]



### imshow w vs ka with colorbar for power
data_bins = power_w_ka_bins.T  # Transpose to have w on y-axis and ka on x-axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(data_bins, origin='lower', aspect='auto', extent=[ka_bins[0], ka_bins[-1], w_bins[0], w_bins[-1]], cmap='magma')

plt.show()






    




