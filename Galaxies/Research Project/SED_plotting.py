from SED_plotter import plot_sed, check_flux_units
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

### Read in the master catalog
mfile = fits.open('/Volumes/Research/COSMOS-Web/COSMOSWeb_mastercatalog_v1.1.fits')
phot = Table(mfile[1].data)
phys = Table(mfile[2].data)

filedata = [phot, phys]
check_flux_units(filedata[0])

### Ids to plot, read in txt file
ids_to_plot = np.loadtxt('ids_above_color_cut.txt', dtype=int)

for id in tqdm(ids_to_plot):
    plot_sed(filedata = filedata, idx = id, yaxis = 'flux', savepath = f'/Users/teddysmith/CU Boulder/1st Year/Galaxies/Project/SED Plots/{id}')




