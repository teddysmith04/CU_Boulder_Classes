import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fftn, fftshift

### Compute the 3D power spectrum of the data cube
### Will result in temporal frequency and two wavenumbers
#### Eventiually want to plot k1 vs k2 with a colorbar for temporal frequency

## Load in data
filepath = '/Users/teddysmith/CU Boulder/1st Year/MathMeth/AMP HWs/Codes/HW5 Codes/hmi.rdVtrack_fd15.2121.225.225.0.-22.5.-00.0.V.fits'
file = fits.open(filepath)

data = file[0].data


## compute 3D fft
fft_data = fftn(data, workers = -1)
fft_w = np.fft.fftfreq(data.shape[0])

### Extract power spectrum 
power_spectrum = np.abs(fft_data)**2
power_spectrum = fftshift(power_spectrum, axes=(0, 1, 2))
log_power_spectrum = np.log10(power_spectrum)

### Show random slices of the data cube power spectrum
insts = np.arange(0, 2304, 400)

for t, w in zip(insts, fft_w[insts]):
    plt.imshow(log_power_spectrum[t], cmap='gray')
    plt.title(f'w = {w:.3f}')
    plt.colorbar(label='log10(Power Spectrum)')
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


### Show slice of w vs kx for different ky
kys = np.fft.fftfreq(data.shape[1])

insts_ky = np.arange(0, 225, 50)
for t, ky in zip(insts, kys[insts_ky]):
    plt.imshow(log_power_spectrum[:, t, :], cmap='gray')
    plt.title(f'ky = {ky:.3f}')
    plt.colorbar(label='log10(Power Spectrum)')
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()




