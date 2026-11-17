import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from IPython.display import IFrame
from astropy.io import fits
import h5py as h5

mfile = fits.open('/Volumes/Research/COSMOS-Web/COSMOSWeb_mastercatalog_v1.1.fits')

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
cat_photom = Table(mfile[1].data)
cat_lephare = Table(mfile[2].data)

# First, define necessary functions

## Define the filter wavenelghts
filters_lambda_eff_bb = {
'cfht-u'      :       [0.3690, 0.0456],
'hsc-g'       :       [0.4851, 0.1194],
'hsc-r'       :       [0.6241, 0.1539],
'hsc-i'       :       [0.7716, 0.1476],
'hsc-z'       :       [0.8915, 0.0768],
'hsc-y'       :       [0.9801, 0.0797],
'uvista-y'    :       [1.0222, 0.0919],
'uvista-j'    :       [1.2555, 0.1712],
'uvista-h'    :       [1.6497, 0.2893],
'uvista-ks'   :       [2.1577, 0.2926],
'hst-f814w'   :       [0.8068, 0.1610],
'f115w'       :       [1.1622, 0.2646],
'f150w'       :       [1.5106, 0.3348],
'f277w'       :       [2.8001, 0.6999],
'f444w'       :       [4.4366, 1.1109],
'f770w'       :       [7.7108, 2.0735]}

filters_lambda_eff_nb = {
'sc-ib427'    :       [0.4264, 0.0207],
'sc-ia484'    :       [0.4851, 0.0228],
'sc-ib505'    :       [0.5064, 0.0231],
'sc-ia527'    :       [0.5262, 0.0242],
'sc-ib574'    :       [0.5766, 0.0272],
'sc-ia624'    :       [0.6234, 0.0301],
'sc-ia679'    :       [0.6783, 0.0336],
'sc-ib709'    :       [0.7075, 0.0316],
'sc-ia738'    :       [0.7363, 0.0323],
'sc-ia767'    :       [0.7687, 0.0364],
'sc-ib827'    :       [0.8246, 0.0344],
'sc-nb711'    :       [0.7120, 0.0073],
'sc-nb816'    :       [0.8150, 0.0120],
'hsc-nb0816'   :       [0.8168, 0.0110],
'hsc-nb0921'   :       [0.9205, 0.0133],
'hsc-nb1010'  :       [1.0100, 0.0094]
}


## Define a function that loads LePHARE SED
def load_lephare_sed(source_id, path="/Volumes/Research/COSMOS-Web"):
    ''' Reads the LePHARE SED for a source given its id. Takes only the best-fit SED.
    '''
    from astropy import units as u
    def check_strictly_increasing(arr):
        # Check if each element is smaller than the next one
        is_increasing = all(x < y for x, y in zip(arr, arr[1:]))
        
        # Find the indices where the array is not strictly increasing
        non_increasing_indices = [i for i in range(len(arr) - 1) if arr[i] >= arr[i + 1]]
        
        return is_increasing, non_increasing_indices
        
    # Path to the spectral file
    spec_filename = f'{path}/COSMOSWeb-LePhare-SEDs_v1.1.h5'

    try:
        specfile = h5.File(spec_filename, 'r')
        print(specfile['ebv'].keys())

        ## make sure to select only the best-fit galaxy spectrum 
        is_increasing, non_increasing_indices = check_strictly_increasing(specfile[:,0])
        if is_increasing == False:
            specfile = specfile[:non_increasing_indices[0], :non_increasing_indices[0]]
            
        # Convert lists to numpy arrays
        lambda_um = specfile[:,0]/10000 #converting to microns
        mag_ab = specfile[:,1]
        
        
        # Convert magnitude (AB) to flux density (F_nu) in microJanskys
        sed_fnu = (mag_ab * u.ABmag).to(u.uJy).value
        
        return lambda_um, sed_fnu
    except Exception as e:
        print(f"Error processing source {source_id}: {e}")
        return np.linspace(0.1,10, 1000), np.zeros((1000))
    

# Define a function that loads the CIGALE SED
def load_cigale_sed(source_id, path="."):
    '''The CIGALE SEDs are saved in the folder P1 for sources with z<5 and P2 for sources with z>5.
    '''
    try:
        with fits.open(f'{path}/CIGALE_SEDs_v1/P1/{source_id}.0_best_model.fits') as hdu:
            cigale_model = hdu[1].data
    except:
        with fits.open(f'{path}/CIGALE_SEDs_v1/P2/{source_id}.0_best_model.fits') as hdu:
            cigale_model = hdu[1].data
    return cigale_model

# Define a function to get the PDF(z) from LePHARE for a given source id
# def load_lephare_pdfz(source_id, pdz=lephare_pdfz):
#     '''Function to get the PDF(z) from LePHARE for a given source id. 
#     The PDF(z) are stored in a pickle file which is a dictionary, where the index correponds to the source id
#     The first array is the redshift array scond array is the Bayesian PDFz, the third one is the likelihood
#     '''

#     return pdz[source_id]
    

def plot_source_photometry_sed(source_id):
    ''' Takes an id of an individual source and plots the flux for each band and the best-fit SED from both LePHARE and CIGALE
    '''
    # take the source from the catalog
    source_cond = np.where(cat_photom['id']==source_id)

    photometry = []
    photometry_err = []
    band_wave = []
    band_wave_width = []

    plt.figure(figsize=(7,5), dpi=150)
    
    # loop over each band to load the broad band photometry
    for band in filters_lambda_eff_bb:
        photometry.append(cat_photom[source_cond][f'flux_model_{band}'][0])
        photometry_err.append(cat_photom[source_cond][f'flux_err-cal_model_{band}'][0])
        band_wave.append(filters_lambda_eff_bb[band][0])
        band_wave_width.append(filters_lambda_eff_bb[band][1])

    photometry = np.asarray(photometry)
    photometry_err = np.asarray(photometry_err)
    band_wave = np.asarray(band_wave)
    band_wave_width = np.asarray(band_wave_width)
    mask = (photometry/photometry_err>1)
    # plot broad band photometry
    plt.errorbar(band_wave[mask], photometry[mask], xerr=band_wave_width[mask], yerr=photometry_err[mask], fmt='o', ms=5, color='navy', label='Broad bands')
    plt.errorbar(band_wave[~mask], photometry[~mask], yerr=photometry_err[~mask], fmt='o', ms=5, color='navy', label='S/N<1', alpha=0.2)

    photometry = []
    photometry_err = []
    band_wave = []
    band_wave_width = []
    # loop over each band to load the medium/narrow band photometry
    for band in filters_lambda_eff_nb:
        photometry.append(cat_photom[source_cond][f'flux_model_{band}'][0])
        photometry_err.append(cat_photom[source_cond][f'flux_err-cal_model_{band}'][0])
        band_wave.append(filters_lambda_eff_nb[band][0])
        band_wave_width.append(filters_lambda_eff_nb[band][1])

    photometry = np.asarray(photometry)
    photometry_err = np.asarray(photometry_err)
    band_wave = np.asarray(band_wave)
    band_wave_width = np.asarray(band_wave_width)
    mask = (photometry/photometry_err>1)
    # plot narrow/medium band photometry
    plt.errorbar(band_wave[mask], photometry[mask], xerr=band_wave_width[mask], yerr=photometry_err[mask], fmt='o', alpha=0.6, ms=3, lw=1, color='navy', label='Narrow/medium bands')
    plt.errorbar(band_wave[~mask], photometry[~mask], yerr=photometry_err[~mask], fmt='o', ms=3, lw=1, color='navy', alpha=0.2)

    ### write properties
    idd = cat_photom[source_cond]['id'][0]
    z = cat_lephare[source_cond]['zfinal'][0]
    mass = cat_lephare[source_cond]['mass_med'][0]
    ssfr = cat_lephare[source_cond]['ssfr_med'][0]
    nuvr = cat_lephare[source_cond]['mabs_nuv'][0]-cat_lephare[source_cond]['mabs_r'][0]
    rj   = cat_lephare[source_cond]['mabs_r'][0]-cat_lephare[source_cond]['mabs_j'][0]
    
    plt.text(x=0.33,y=0.8,s=
                fr'id = {idd}'
                '\n'
                fr'$z$ = {z:.2f}'
                '\n'
                r'log$M_{\star}/{\rm M_{\odot}} = $'+f'{mass:.2f}'
                '\n'
                r'log sSFR$/$yr$^{-1} = $'+f'{ssfr:.2f}'
                '\n'
                fr'$NUV-r$ = {nuvr:.2f}'
                '\n'
                fr'$r-J$ = {rj:.2f}'
             ,fontsize=10) #,bbox=dict(facecolor='white'))

    # load and plot LePHARE SED
    lambda_lephare, sed_lephare = load_lephare_sed(source_id) 
    plt.plot(lambda_lephare, sed_lephare, color='tomato', label='LePHARE SED', lw=2, zorder=-10)

    # # load and plot CIGALE SED
    # try:
    #     cigale_model = load_cigale_sed(source_id)
    #     plt.plot(cigale_model['wavelength']/1000, (cigale_model['Fnu']*1.0e3), 
    #         lw=2, alpha=0.7, color='goldenrod', label='CIGALE SED', zorder=-11)
    # except:
    #     print('Can not find CIGALE output for this source')

    plt.xlim(0.3, 10)
    plt.ylim(0.001, 10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel(r'Flux [$\mu$Jy]')
    plt.legend(loc=4)
    plt.minorticks_on()
    plt.show()


    # Plot the PDF(z)
    # pdfz = load_lephare_pdfz(source_id)
    # plt.figure(figsize=(6,2), dpi=150)
    # plt.plot(pdfz[0,:], pdfz[1,:]/np.max(pdfz[1,:]), label='Bayesian', alpha=0.6)
    # plt.plot(pdfz[0,:], pdfz[2,:]/np.max(pdfz[2,:]), label='Profile', alpha=0.6)
    # plt.xlabel(r'Redshift')
    # plt.ylabel(r'$P(z)$ norm. to max=1')
    # plt.legend()
    # plt.minorticks_on()
    # plt.show()

    
    return


load_lephare_sed(564339)