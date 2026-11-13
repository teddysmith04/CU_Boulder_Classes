import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fftn, fftshift, fftfreq
from scipy.signal.windows import hann

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


## Plot power vs temporal frequency for selected pixel
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(w, power_0, color='blue', lw=1.5)
ax.set(
xlabel='Temporal Frequency [mHz]',
       ylabel='Power [arb. units]',
    #    title='Power Spectrum at k_x = k_y = 0'

)
plt.show()




