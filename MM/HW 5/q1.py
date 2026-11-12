from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt


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

## __________________________________________________________________________________________
## Need to read in a 2304 x 384 x 384 data cube fits file. The dimensions are  time x X x Y. 
## Extract some images and show tham as a funciton of X and Y, then slice along a row or collumn and plot line plots of the doppler velocity signal as a function of X or Y
## __________________________________________________________________________________________


### Load in Data file 
filepath = '/Users/teddysmith/CU Boulder/1st Year/MathMeth/AMP HWs/Codes/HW5 Codes/hmi.rdVtrack_fd15.2121.225.225.0.-22.5.-00.0.V.fits'

## Read in data and read header
hdu = fits.open(filepath)


### Idk why this works
data = hdu[0].data

## Pull out like 10 images and show them as a function of X and Y

inst = np.arange(0, 2304, 400)

vmax = np.nanpercentile(np.abs(data[inst]), 99)
vmin = -vmax
 
fig, axes = plt.subplots(2, 3, figsize=(9, 6),
                         gridspec_kw={'hspace': 0.0, 'wspace': 0.0})


for ax, t in zip(axes.flat, inst):
    im = ax.imshow(data[t], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    # Small label so each panel is identifiable without axis ticks
    ax.text(0.03, 0.95, f't = {t}', transform=ax.transAxes, color='white',
            va='top', ha='left', fontsize=12)
 
### Colorbar axis spanning the full height of the image grid
bottom = axes[1, 0].get_position().y0
top = axes[0, 0].get_position().y1
cax = fig.add_axes([0.90, bottom, 0.025, top - bottom])
fig.colorbar(im, cax=cax, label='Doppler Velocity [m/s]', pad = 0)
 
plt.show()
plt.cla()
plt.clf()
plt.close()


### For each inst, slice along a random row and plot the doppler velocity signal as a function of X
np.random.seed(111)
row_random = np.random.randint(0, data.shape[1])
 
fig2, axs2 = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True,
                          gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
 
for ax, t in zip(axs2.flat, inst):
    ax.plot(data[t, row_random, :], color='blue', lw=1)
    ax.text(0.03, 0.95, f't = {t}', transform=ax.transAxes, va='top', ha='left', fontsize=12)
 
fig2.supxlabel('X [pixels]')
fig2.supylabel('Doppler Velocity [m/s]')

plt.show()
plt.cla()
plt.clf()
plt.close()

##### Calculate the RMS velocity of each image
data_inst = data[inst,:,:]
#flatten
data_flat = data_inst.reshape(data_inst.shape[0], -1)

for i in range(5):
    rms_velocity = np.sqrt(np.nanmean(data_flat[i]**2))
    print(f"RMS velocity for image {inst[i]}: {rms_velocity/1000:.2f} km/s")






 





