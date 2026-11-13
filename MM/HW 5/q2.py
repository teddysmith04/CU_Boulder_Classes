import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fftn, fftshift, fftfreq
from scipy.signal.windows import hann
from matplotlib.ticker import MaxNLocator

### Compute the 3D power spectrum of the data cube
### Will result in temporal frequency and two wavenumbers
#### Eventiually want to plot k1 vs k2 with a colorbar for temporal frequency
DT = 45.0  # time cadence of the data cube [s]
## Load in data
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
 
 
def log_power(img):
    """log10 of a 2D slice, computed only on the slice (not the full cube)."""
    return np.log10(img + 1e-30)

### Show random slices of the data cube power spectrum
def shared_grid(slices, labels, extent, xlabel, ylabel, ylim=None):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    # fig.subplots_adjust(left=0.09, right=0.86, bottom=0.11, top=0.98)
 
    # One common color range for every panel
    vmin = min(np.percentile(s, 1) for s in slices)
    vmax = max(np.percentile(s, 99.9) for s in slices)
 
    for ax, img, lab in zip(axes.flat, slices, labels):
        im = ax.imshow(img, origin='lower', extent=extent, aspect='auto',
                       cmap='gray', vmin=vmin, vmax=vmax)
        ax.text(0.04, 0.95, lab, transform=ax.transAxes, color='white',
                va='top', ha='left', fontsize=12)
        # Trim the outermost tick labels so they don't collide where panels touch
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
 
    if ylim is not None:
        axes[0, 0].set_ylim(*ylim)
 
    # Colorbar axis spanning the full height of the grid
    bottom = axes[1, 0].get_position().y0
    top = axes[0, 0].get_position().y1
    cax = fig.add_axes([0.88, bottom, 0.025, top - bottom])
    fig.colorbar(im, cax=cax, label=r'$\log_{10}$(Power)')
 
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    return fig
 
 
## __________________________________________________________________________________________
## Figure 1: ky vs kx slices at different temporal frequencies (positive frequencies only)
## __________________________________________________________________________________________
w_idx = it0 + np.array([50, 150, 300, 450, 600, 900])   # edit these to choose your slices
 
slices = [log_power(power[i]) for i in w_idx]
labels = [f'w = {w[i]:.2f} mHz' for i in w_idx]
 
shared_grid(slices, labels,
            extent=[kx[0], kx[-1], ky[0], ky[-1]],
            xlabel=r'$k_x$ [cycles / pixel]',
            ylabel=r'$k_y$ [cycles / pixel]')
plt.show()
 
## __________________________________________________________________________________________
## Figure 2: w vs kx slices at different ky (positive frequencies only)
## __________________________________________________________________________________________
ky_idx = iy0 + np.array([0, 20, 40, 80, 120, 160])      # edit these to choose your slices
 
slices = [log_power(power[it0:, i, :]) for i in ky_idx]
labels = [rf'$k_y$ = {ky[i]:.3f}' for i in ky_idx]
 
shared_grid(slices, labels,
            extent=[kx[0], kx[-1], w[it0], w[-1]],
            xlabel=r'$k_x$ [cycles / pixel]',
            ylabel='w [mHz]',
            # ylim=(0, 8)) 
  )  # zoom on the low-frequency band; remove to see everything
plt.show()




