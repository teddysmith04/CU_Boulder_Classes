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


fig, ax = plt.subplots(2, 3, figsize=(8, 6), gridspec_kw={'hspace': 0.0, 'wspace': 0.0})

for i in range(len(inst)):
    data_i = data[inst[i], :, :]
    if i <= 2:
        im = ax[0][i].imshow(data_i, cmap='gray')
        
    else:
        im = ax[1][i-4].imshow(data_i, cmap='gray')
        
# fig.colorbar(im, ax=ax, pad = 0, label='Doppler Velocity [m/s]', shrink=0.5)
plt.show()




