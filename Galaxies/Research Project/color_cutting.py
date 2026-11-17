import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM



plt.rcParams.update({
    'font.size': 14,
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

#### Mster file path
mfile_path = '/Volumes/Research/COSMOS-Web/COSMOSWeb_mastercatalog_v1.1.fits'

### Open the master catalog
mfile = fits.open(mfile_path)

### Open file and get good ids
phot = mfile[1]
phys = mfile[2]
flags = phot.data.field('warn_flag')
mass = phys.data.field('mass_minchi2')
z = phys.data.field('zfinal')
MIRI_flux = phot.data.field('flux_auto_f770w')

good_indicies = np.where((z > 4) & (mass > -50) & (flags == 0) & (~np.isnan(MIRI_flux)))[0]

## Get Id, F277W flux, F277W flux error, F444W flux, F444W flux error, zfinal,
ids = phot.data.field('id')[good_indicies]
zs = phys.data.field('zfinal')[good_indicies]

F277W_flux = phot.data.field('flux_auto_f277w')[good_indicies] * 1e-29 ## in erg / s cm Hz
F277W_err = phot.data.field('flux_err_auto_f277w')[good_indicies] * 1e-29 ## in erg / s cm Hz

F444W_flux = phot.data.field('flux_auto_f444w')[good_indicies] * 1e-29 ## in erg / s cm Hz
F444W_err = phot.data.field('flux_err_auto_f444w')[good_indicies] * 1e-29 ## in erg / s cm Hz

MIRI_flux = phot.data.field('flux_auto_f770w')[good_indicies] * 1e-29 ## in erg / s cm Hz


### Turn flux into Mag
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
distances = cosmo.luminosity_distance(zs).to('cm').value

luminosity_F277W = F277W_flux * 4 * np.pi * distances**2
luminosity_F444W = F444W_flux * 4 * np.pi * distances**2

### Turn to M
Msun = 4.74
Lsun = 3.828e33  # erg/s
F277W_Mag = -2.5 * np.log10(luminosity_F277W /Lsun) + Msun
F444W_Mag = -2.5 * np.log10(luminosity_F444W /Lsun) + Msun

color = (F277W_Mag - F444W_Mag)

#### Plot color vs F444W Mag with each point colored on a colorbar representing z
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(F444W_Mag, color, c=zs, cmap='viridis', s=20, alpha=0.7, edgecolors='w', rasterized=True)
ax.axhline(y=1.5, color='r', linestyle='--', label='Color Cut = 1.5')
ax.set(
    xlabel='F444W Mag',
    ylabel='F277W Mag - F444W Mag',
    ylim = (0, 12)
)
cbar = plt.colorbar(scatter, ax=ax, pad = 0)
cbar.set_label('Redshift (z)')
plt.legend()
# plt.show()

### Get ID of the biggest one bc its cool 
id_max = ids[np.nanargmax(color)]

## 2nd most red object
valid_indices = np.where(~np.isnan(color))[0]

# Sort valid indices based on their corresponding array values in descending order
sorted_valid_indices = valid_indices[np.argsort(color[valid_indices])[::-1]]

# The 2nd highest non-NaN index is at position 1
second_highest_index = sorted_valid_indices[50]


print(ids[second_highest_index], color[second_highest_index], zs[second_highest_index])




### Look at mr107 (what im calling id  = 107)

