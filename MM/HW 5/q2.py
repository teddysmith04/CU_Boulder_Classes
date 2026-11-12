import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fftn

### Compute the 3D power spectrum of the data cube
### Will result in temporal frequency and two wavenumbers
#### Eventiually want to plot k1 vs k2 with a colorbar for temporal frequency

## Load in data
filepath = '/Users/teddysmith/CU Boulder/1st Year/MathMeth/AMP HWs/Codes/HW5 Codes/hmi.rdVtrack_fd15.2121.225.225.0.-22.5.-00.0.V.fits'
file = fits.open(filepath)

data = file[0].data


## compute 3D fft
fft_data = fftn(data, workers = -1)

### Extract power spectrum 
power_spectrum = np.abs(fft_data)**2
log_power_spectrum = np.log10(power_spectrum)

### Show random slices of the data cube power spectrum
insts = np.arange(0, 2304, 400)

for t in insts:
    plt.imshow(log_power_spectrum[t], cmap='gray')
    plt.title(f't = {t}')
    plt.colorbar(label='log10(Power Spectrum)')
    plt.show()



