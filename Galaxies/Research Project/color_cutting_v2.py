from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from image_look import plot_source_photometry_sed


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

mfile_path = '/Volumes/Research/COSMOS-Web/COSMOSWeb_mastercatalog_v1.1.fits'
mfile = fits.open(mfile_path)

phot = Table(mfile[1].data)
phys = Table(mfile[2].data)


condition_clean = np.logical_and.reduce((
    phot['warn_flag']==0, # No warning flag
    np.abs(phot['mag_auto_f444w'])<30, # Remove very faint objects
    phot['flag_star_hsc']==0, # Remove objects in HSC star mask area
    phys['zfinal']>7, # Remove objects with zfinal < 4
    np.abs(phot['mag_auto_f277w'])<30, # Remove very faint objects
    phot['flux_auto_f277w']>0, # Remove objects with negative flux
    phot['flux_auto_f444w']>0, # Remove objects with negative flux
    # phys['flag_chandra'] == 0, ## Remove objects without Chandra flag
    phys['chi_star'] > 10.0, ## Remove objects with chi_star < 10
    np.abs(phys['zfinal'] - phys['zpdf_l68']) < 1.0, ## Remove objects with zfinal and zpdf_l68 differing by more than 1
    phot['snr_f277w'] > 5, ## Remove objects with SNR < 5 in F277W
    phot['snr_f444w'] > 5, ## Remove objects with SNR < 5 in F444W
    ### Select masses > log (M) > 9.5
    phys['mass_minchi2'] > 9.5,
))

F444W_flux = phot['flux_auto_f444w'][condition_clean] * 1e-29 ## in erg / s cm Hz
F277W_flux = phot['flux_auto_f277w'][condition_clean] * 1e-29 ## in erg / s cm Hz

zs = phys['zfinal'][condition_clean]
ids = phot['id'][condition_clean]
agn_chi2 = phys['chi2_agn'][condition_clean]   

AGN_threshold = 100.0

AGN_condition = np.logical_and.reduce((
    phys['chi2_agn'][condition_clean] > AGN_threshold
))






cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
distances = cosmo.luminosity_distance(zs).to('cm').value

luminosity_F277W = F277W_flux * 4 * np.pi * distances**2
luminosity_F444W = F444W_flux * 4 * np.pi * distances**2

### Turn to M
Msun = 4.74
Lsun = 3.828e33  # erg/s
F277W_Mag = phot['mag_auto_f277w'][condition_clean]
F444W_Mag = phot['mag_auto_f444w'][condition_clean]

F227W_agnmag = phot['mag_auto_f277w'][AGN_condition]
F444W_agnmag = phot['mag_auto_f444w'][AGN_condition]

color = (F277W_Mag - F444W_Mag)
agn_color = (F227W_agnmag - F444W_agnmag)

#### Plot color vs F444W Mag with each point colored on a colorbar representing z
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(F444W_Mag, color, c=zs, cmap='viridis', s=40, alpha=0.7, edgecolors='k')
scatter_agn = ax.scatter(F444W_agnmag, agn_color, c='r', marker = '*', s=40, alpha=1, edgecolors='k', label='AGN Candidates')
ax.axhline(y=1.5, color='r', linestyle='--', label='Color Cut = 1.5')
ax.set(
    xlabel='F444W Mag',
    ylabel='F277W Mag - F444W Mag',
    # ylim = (0, 8)
)
cbar = plt.colorbar(scatter, ax=ax, pad = 0)
cbar.set_label('Redshift (z)')
plt.legend()
plt.savefig('/Users/teddysmith/CU Boulder/1st Year/Galaxies/Proposal/figures', bbox_inches='tight', dpi=300)
plt.show()

### Get Ids above color cut
Indexs_above_color_cut = np.where(color > 1.5)[0]
ids_above_color_cut = ids[Indexs_above_color_cut]

del mfile
del phot
del phys

print('AGN Chi2 for ids above color cut:')
print(agn_chi2[Indexs_above_color_cut])


### Save ids as txt file
# np.savetxt('ids_above_color_cut.txt', ids_above_color_cut, fmt='%d')


