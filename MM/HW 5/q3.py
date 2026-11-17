import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fftn, fftshift, fftfreq
from scipy.signal.windows import hann
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

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

filepath = '/Users/teddysmith/CU Boulder/1st Year/MathMeth/HWs/Codes/HW5 Codes/hmi.rdVtrack_fd15.2121.225.225.0.-22.5.-00.0.V.fits'
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

n_ka = 50
ka_edges = np.linspace(0.0, 0.5, n_ka + 1)          # 0 -> Nyquist (full annuli only)
ka_centers = 0.5 * (ka_edges[1:] + ka_edges[:-1])

kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')   # shape (nx, ny), matches power[i]
ka_flat = np.hypot(kx_grid, ky_grid).ravel()

# Bin indices depend only on k, so compute once, outside the loop
ka_idx = np.digitize(ka_flat, ka_edges) - 1
valid = (ka_idx >= 0) & (ka_idx < n_ka)           # drops the corners beyond Nyquist
ka_idx = ka_idx[valid]
counts = np.bincount(ka_idx, minlength=n_ka)      # pixels per annulus

power_w_ka = np.zeros((nt, n_ka))                 # (w, ka)
for i in range(nt):
    sums = np.bincount(ka_idx, weights=power[i].ravel()[valid], minlength=n_ka)
    power_w_ka[i] = sums / np.maximum(counts, 1)  # mean power per annulus

### Positive frequencies only (the spectrum of a real signal is symmetric in w)
pos = w >= 0
img = power_w_ka[pos]
w_pos = w[pos]

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(img, origin='lower', aspect='auto', cmap='magma',
               extent=[ka_edges[0], ka_edges[-1], w_pos[0], w_pos[-1]],
               norm=LogNorm(vmin=np.percentile(img[img > 0], 1), vmax=img.max()))
ax.set_xlabel(r'$k_a$ [cycles / pixel]')
ax.set_ylabel(r'$\omega$ [mHz]')
fig.colorbar(im, ax=ax, label='Power', pad = 0.0)
plt.savefig('/Users/teddysmith/CU Boulder/1st Year/MathMeth/HWs/HW5/ka_power_spectrum.pdf', bbox_inches='tight')
plt.show()
plt.cla()
plt.clf()
plt.close()

### Pick out a slice of ka = 0.1 and plot the power as a function of w compare to plot from single pixel kx = 0.1, ky = 0

ka_slice = 0.1
ka_idx = np.argmin(np.abs(ka_centers - ka_slice))

kx_pix = 0.1
kx_idx = np.argmin(np.abs(kx - kx_pix))
ky = 0.0
ky_idx = np.argmin(np.abs(ky - ky))

fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
axs[0].plot(w_pos, power_w_ka[pos, ka_idx], color='k', lw=1)
axs[0].set_xlabel(r'$\omega$ [mHz]')
axs[0].set_ylabel('Power')
axs[0].set_yscale('log')
axs[1].plot(w_pos, power[pos, ky_idx, kx_idx], color='k', lw=1)
axs[1].set_xlabel(r'$\omega$ [mHz]')
axs[1].set_yscale('log')

plt.savefig('/Users/teddysmith/CU Boulder/1st Year/MathMeth/HWs/HW5/ka_vs_pixel_power_spectrum.pdf', bbox_inches='tight')

plt.show()
plt.cla()
plt.clf()
plt.close()










    




