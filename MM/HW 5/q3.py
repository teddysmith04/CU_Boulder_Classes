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
data *= hann(nx).astype(np.float32)[None, :, None]
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


## Plot power vs temporal frequency for selected pixel          #|
# fig, ax = plt.subplots(figsize=(8, 5))                        #| THIS WORKS
# ax.plot(w, power_0, color='blue', lw=1.5)                     #|
# ax.set(
# xlabel='Temporal Frequency [mHz]',                                
#        ylabel='Power [arb. units]',
    #    title='Power Spectrum at k_x = k_y = 0'

# )
# plt.show()

### ___________________________________________________________________________________________
### Now the hard part. Since power is isotropic, we can look at some wavenumber ka = sqrt(kx^2 + ky^2). Pick a representative range that spans kx, ky for ka, 
### and then for each ka, bin the power in around some ka - delta ka and ka + delta ka. Integrate the power in that bin and plot the integrated paowr per bin vs frequency. 
### ___________________________________________________________________________________________

### New idea, screw using ka bin kx and ky simultaneously with the same delta k. Then integrate power for some frequency w within the kx ky bin 

power_in_bins = {} ## heres my idea, each entry in this dict will be like bin 1: then an array of power in bin 1 for every frequency

kx_max, kx_min, ky_max, ky_min = kx.max(), kx.min(), ky.max(), ky.min()

## Define bins 
dk = 0.1
# print((kx_max - kx_min)//dk) #WILL return 9 bins with dk = 0.1

for kx_bin, ky_bin in zip(np.arange(kx_min, kx_max, dk), np.arange(ky_min, ky_max, dk)):
    kx_bin_min, kx_bin_max = kx_bin, kx_bin + dk
    ky_bin_min, ky_bin_max = ky_bin, ky_bin + dk

    ## find the indices of the power array that correspond to the bin
    kx_indices = np.where((kx >= kx_bin_min) & (kx < kx_bin_max))[0]
    ky_indices = np.where((ky >= ky_bin_min) & (ky < ky_bin_max))[0]

    ## integrate power in the bin for each frequency w
    power_in_bins[f"bin_{kx_bin:.2f}_{ky_bin:.2f}"] = np.zeros(len(w))
    for i in range(len(w)):
        power_in_bins[f"bin_{kx_bin:.2f}_{ky_bin:.2f}"][i] = np.sum(power[i, ky_indices[:, None], kx_indices[None, :]])

### Plot for one bin
# print(f"Power in bins: {power_in_bins.keys()}")

fig, axs = plt.subplots(2, 5, figsize=(15, 6), gridspec_kw={'hspace': 0, 'wspace': 0})

for ax, bin_key in zip(axs.flat, power_in_bins.keys()):
    ax.plot(w, power_in_bins[bin_key], color='blue', lw=1.5)
    ax.text(0.03, 0.95, bin_key, transform=ax.transAxes, color='black', va='top', ha='left', fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))


plt.show()
   







    




