import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import matplotlib.pyplot as plt

from scipy.integrate import quad

from colossus.cosmology import cosmology
from colossus.lss import mass_function


## Define a cosmo
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


# Master catalog path
mfile_path = '/Volumes/Research/COSMOS-Web/COSMOSWeb_mastercatalog_v1.1.fits'

## Read in data
mfile = fits.open(mfile_path)
phot = Table(mfile[1].data)
phys = Table(mfile[2].data)

condition_clean = np.logical_and.reduce((
    phot['warn_flag']==0, # No warning flag
    np.abs(phot['mag_auto_f444w'])<30, # Remove very faint objects
    phot['flag_star_hsc']==0, # Remove objects in HSC star mask area
    # phys['zfinal']>7, # Remove objects with zfinal < 4
    np.abs(phot['mag_auto_f277w'])<30, # Remove very faint objects
    phot['flux_auto_f277w']>0, # Remove objects with negative flux
    phot['flux_auto_f444w']>0, # Remove objects with negative flux
    phys['chi_star'] > 10.0, ## Remove objects with chi_star < 10
    np.abs(phys['zfinal'] - phys['zpdf_l68']) < 1.0, ## Remove objects with zfinal and zpdf_l68 differing by more than 1
    phot['snr_f277w'] > 5, ## Remove objects with SNR < 5 in F277W
    phot['snr_f444w'] > 5, ## Remove objects with SNR < 5 in F444W
    ### Select masses > log (M) > 9.5
    phys['mass_minchi2'] > 9.5,

))

## Make arrays for masses
masses = phys['mass_minchi2'][condition_clean]

### Bin the data by redshift 4-4.5
zbin = np.arange(4, 10, 0.5)
masses_binned = np.array([])
number_of_galaxies_binned = np.array([])
bin_volumes = np.array([])

ns = []
nerr = []

for i in range(len(zbin)-1):
    zmin = zbin[i]
    zmax = zbin[i+1]
    condition_z = np.logical_and(phys['zfinal'][condition_clean] >= zmin, phys['zfinal'][condition_clean] < zmax)
    ### Append arrays for masses in each bin
    masses_binned = np.append(masses_binned, [masses[condition_z]])
    number_in_bin = len(masses[condition_z])
    number_of_galaxies_binned = np.append(number_of_galaxies_binned, number_in_bin)

    ### Calculate volume of bin
    dmin = cosmo.comoving_distance(zmin).value
    dmax = cosmo.comoving_distance(zmax).value
    
    ### Volume of space from a survey 0.54 deg^2
    survey_area_deg2 = 0.54
    survey_area_str = survey_area_deg2 * (np.pi/180)**2  # Convert to steradians
    bin_volume = survey_area_str/3 * (dmax**3 - dmin**3)  # Volume in Mpc^3
    bin_volumes = np.append(bin_volumes, bin_volume)

    ## Compune poisson errors in each bin
    N_err = np.sqrt(number_in_bin)


    ## Compute number density in each bin and error
    n_bin = number_in_bin / bin_volume
    n_bin_err = N_err / bin_volume

    ns.append(n_bin)
    nerr.append(n_bin_err)


log_n = np.log10(ns)
# log_nerr = np.log10(nerr)




## Compurte poisson errors on number densities

### PLOT log number density vs redshift
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(zbin[:-1] + 0.25, ns, yerr=nerr, fmt='o', color='k', label='Number Density', capsize=5)
ax.set(
xlabel='Redshift (z)', ylabel='log(Number Density [Mpc^-3])', yscale = 'log'
)
ax.text(0.9, 0.9, r'$M_* > 10^{9.5} M_\odot$', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

#### Compute theoretical number densities
plt.show()
plt.cla()
plt.clf()
plt.close()


##### HOW DO I COMPUTE THEORETICAL NUMBER DENSITIES FOR HALO MASS FUNCTION?
#### I WANT TO PLOT ON THE SAME PLOT LIKE IN CHWOWORSKY+25
##### M_star = f_baryon * epsilon * M_halo

f_baryon = 0.16  # Cosmic baryon fraction
