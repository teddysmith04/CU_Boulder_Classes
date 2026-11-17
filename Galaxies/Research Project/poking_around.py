import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table

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

#### Figure out whats in each file
phot = Table(mfile[1].data)
phys = Table(mfile[2].data)



### Clean data
condition_clean = np.logical_and.reduce((
    phot['warn_flag']==0, # No warning flag
    np.abs(phot['mag_auto_f444w'])<30, # Remove very faint objects
    phot['flag_star_hsc']==0, # Remove objects in HSC star mask area
    # phys['zfinal']>7.0, # Remove objects with zfinal < 4
    np.abs(phot['mag_auto_f277w'])<30, # Remove very faint objects
    phot['flux_auto_f277w']>0, # Remove objects with negative flux
    phot['flux_auto_f444w']>0, # Remove objects with negative flux
    # phys['chi_star'] > 10.0, ## Remove objects with chi_star < 10
    np.abs(phys['zfinal'] - phys['zpdf_l68']) < 1.0, ## Remove objects with zfinal and zpdf_l68 differing by more than 1
    phot['snr_f277w'] > 5, ## Remove objects with SNR < 5 in F277W
    phot['snr_f444w'] > 5, ## Remove objects with SNR < 5 in F444W
    ### Select masses > log (M) > 9.5
    # phys['mass_minchi2'] > 9.5,
))

zs = phys['zfinal'][condition_clean]
masses = phys['mass_minchi2'][condition_clean]







###Plot flux vs zfinal as a kind of survey plot
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(good_z['zfinal'], good_z['flux_115w'], 'o', markersize=2, alpha=0.5, rasterized = True)
# ax.set(
# xlabel='Redshift (z)', ylabel='Flux (f115w) nJy', title='COSMOS-Web Master Catalog: Flux vs Redshift'
# )
# plt.show()

### Plot mass vs z
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(zs, masses, 'o', markersize=2, alpha=0.3, rasterized=True, color = 'grey')
# for i in ids_with_massive_qsos:
#     ax.text(z[good_indicies][ids[good_indicies] == i], mass[good_indicies][ids[good_indicies] == i], str(i), fontsize=10, color='k')

ax.set(
    xlabel = 'Redshift (z)', ylabel = r'Log Stellar Mass ($M_\odot$)', ylim = (8.5, 12)

)
ax.axvline(x = 4, color = 'k', linestyle = '-')
ax.axhline(y = 10, color = 'k', linestyle = '-')
### Shade only in the region of interest (z > 4, log(M) > 10)
ax.fill_betweenx(y = [10, 12], x1 = 4, x2 = 14, color = 'grey', alpha = 0.2)


#### Now make points special for NIRSpec coverage, MIRI Coverage, and  both with flags
### Flag miri true if phot['flux_auto_f770w'] > 0 and phot['snr_f770w'] > 5
### Flag zspec true in phot['id_specz_khostovan25'] > 0
### Flags, all on the cleaned sample so they line up with zs and masses
def as_bool(x):
    return np.ma.filled(x, False)  # masked/NaN entries -> False

miri_flag = as_bool((phot['flux_auto_f770w'][condition_clean] > 0) &
                    (phot['snr_f770w'][condition_clean] > 5))
zspec_flag = as_bool(phot['id_specz_khostovan25'][condition_clean] > 0)

### Region of interest: shaded region (z > 4), optionally with a mass cut
MASS_MIN = 10.0   # set to 8.5 to cover the whole shaded region
in_region = (zs > 4.0) & (masses > MASS_MIN)

both      = in_region & miri_flag & zspec_flag
miri_only = in_region & miri_flag & ~zspec_flag
spec_only = in_region & ~miri_flag & zspec_flag

for mask, color, marker, label in [
    (miri_only, 'blue',  's', 'MIRI Coverage'),
    (spec_only, 'green', '^', 'NIRSpec Specz'),
    (both,      'red',   '*', 'MIRI + Spec-z'),
]:
    ax.plot(zs[mask], masses[mask], marker, color=color,
            markersize=12 if marker == '*' else 7,
            markeredgecolor='k', markeredgewidth=0.5,
            linestyle='none', label=f'{label} (N={mask.sum()})')

ax.legend(loc='lower right')


plt.savefig('../figures/mass_vs_z.pdf', dpi=300, bbox_inches='tight')
plt.show()




