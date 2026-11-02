import numpy
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=71, Om0=0.3)

z1 = 0.005
z2 = 0.006

size1 = 0.1 * 1/cosmo.arcsec_per_kpc_proper(z1)
size2 = 0.1 * 1/cosmo.arcsec_per_kpc_proper(z2)

print(f"Angular size per kpc at z={z1}: {size1:.4f} arcsec/kpc")
print(f"Angular size per kpc at z={z2}: {size2:.4f} arcsec/kpc")

distance1 = cosmo.luminosity_distance(z1).value
distance2 = cosmo.luminosity_distance(z2).value

print(f"Luminosity distance at z={z1}: {distance1:.2f} Mpc")    
print(f"Luminosity distance at z={z2}: {distance2:.2f} Mpc")