"""
Recreate Fig. 5 of Chworowsky et al. (2024, AJ 168, 113) with COSMOS-Web:
cumulative number density of log10(M*/Msun) > 10 galaxies vs redshift, with and without
"extremely red objects" (EROs, F277W-F444W > 1.5), compared to the cumulative halo mass
function converted to stellar mass with M* = epsilon * f_baryon * M_halo.

Usage:  python baryon_efficiency.py [/path/to/COSMOSWeb_mastercatalog_v1.1.fits]
"""
import sys
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from colossus.cosmology import cosmology
from colossus.lss import mass_function

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MASTER_PATH = (sys.argv[1] if len(sys.argv) > 1
               else '/Volumes/Research/COSMOS-Web/COSMOSWeb_mastercatalog_v1.1.fits')

M_LIM = 10.0                       # log10(M*/Msun) threshold used in Fig. 5
Z_EDGES = np.arange(4, 11, 1.0)    # Delta z = 1 bins, as in the paper
SURVEY_AREA_DEG2 = 0.54            # NIRCam area. Use the *unmasked* area if you can (see notes)
F_BARYON = 0.16                    # cosmic baryon fraction used in the paper
EPS_LIST = [0.05, 0.14, 0.2, 0.3, 0.5, 1.0]   # baryon conversion efficiencies to draw
COLOR_CUT = 1.5                    # F277W - F444W > 1.5 -> ERO (Barro et al. 2024)
N_DRAWS = 500                      # Monte Carlo draws (paper uses 500)
rng = np.random.default_rng(42)

# Cosmology for the *observed* volumes (same as the paper / your original script)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


# ----------------------------------------------------------------------------
# 1. Observed number densities
# ----------------------------------------------------------------------------
def shell_volume(zmin, zmax, area_deg2, cosmo):
    """Comoving volume [Mpc^3] of the redshift shell seen over `area_deg2`."""
    dmin, dmax = cosmo.comoving_distance([zmin, zmax]).value
    omega = area_deg2 * (np.pi / 180.0) ** 2           # steradians
    return omega / 3.0 * (dmax ** 3 - dmin ** 3)


def draw_split_normal(med, s_lo, s_hi, n_draws, rng):
    """
    Draw from a two-piece normal with median `med`, lower width s_lo and upper width s_hi.
    Returns an array of shape (n_draws, len(med)). Used to turn (med, l68, u68) into samples.
    """
    med = np.asarray(med, float)
    s_lo = np.clip(np.nan_to_num(np.asarray(s_lo, float)), 0, None)
    s_hi = np.clip(np.nan_to_num(np.asarray(s_hi, float)), 0, None)
    tot = s_lo + s_hi
    p_lo = np.divide(s_lo, tot, out=np.full_like(tot, 0.5), where=tot > 0)
    g = np.abs(rng.standard_normal((n_draws, med.size)))
    go_low = rng.random((n_draws, med.size)) < p_lo
    return med + np.where(go_low, -g * s_lo, g * s_hi)


def binned_density(z_draw, m_draw, edges, volumes, rng):
    """
    z_draw, m_draw: (n_draws, n_gal) redshift / log-mass draws.
    Counts galaxies with m_draw > M_LIM in each z bin per draw, adds Poisson scatter,
    and returns median and 16/84 percentiles of the number density [Mpc^-3].
    """
    n_bins = len(edges) - 1
    counts = np.empty((z_draw.shape[0], n_bins))
    for k in range(n_bins):
        in_bin = (z_draw >= edges[k]) & (z_draw < edges[k + 1]) & (m_draw > M_LIM)
        counts[:, k] = in_bin.sum(axis=1)
    dens = rng.poisson(counts) / volumes
    med, lo, hi = np.percentile(dens, [50, 16, 84], axis=0)
    return med, lo, hi, np.median(counts, axis=0)


# ----------------------------------------------------------------------------
# 2. Theoretical number densities from the halo mass function
# ----------------------------------------------------------------------------
# Rodriguez-Puebla+16 (the HMF used by Chworowsky+24) is in colossus, calibrated for a
# Planck cosmology and *virial* halo masses. colossus works in Msun/h and (Mpc/h)^-3.
cosmology.setCosmology('planck15')
H_HMF = cosmology.getCurrent().h
LNM = np.linspace(np.log(1e8), np.log(1e17), 600)       # ln(M_halo / [Msun/h])


def halo_n_above(z, model='rodriguezpuebla16', mdef='vir'):
    """n(>M_halo) on the LNM grid at redshift z, in (Mpc/h)^-3 comoving."""
    dndlnM = mass_function.massFunction(np.exp(LNM), z, q_in='M', q_out='dndlnM',
                                        mdef=mdef, model=model)
    # integrate from the high-mass end downwards
    return -cumulative_trapezoid(dndlnM[::-1], LNM[::-1], initial=0.0)[::-1]


def theory_density(z_grid, eps_list, m_lim=M_LIM, f_b=F_BARYON):
    """
    Cumulative number density [Mpc^-3, comoving] of galaxies with M* > 10^m_lim, assuming
    M* = eps * f_b * M_halo, i.e. halos above M_halo,min = 10^m_lim / (eps * f_b).
    Returns dict {eps: array over z_grid}.
    """
    out = {eps: np.empty(len(z_grid)) for eps in eps_list}
    for i, z in enumerate(z_grid):
        n_above = halo_n_above(z)
        for eps in eps_list:
            mh_min = 10 ** m_lim / (eps * f_b)                  # Msun
            ln_mh = np.log(mh_min * H_HMF)                      # Msun/h
            out[eps][i] = np.interp(ln_mh, LNM, n_above) * H_HMF ** 3   # -> Mpc^-3
    return out


def eps_needed(n_obs, z, eps_grid=np.logspace(-2, 0.5, 300)):
    """Efficiency at which the model reproduces n_obs at redshift z (NaN if out of range)."""
    n_th = np.array([theory_density([z], [e])[e][0] for e in eps_grid])   # increases with eps
    return np.interp(n_obs, n_th, eps_grid, left=np.nan, right=np.nan)


# ----------------------------------------------------------------------------
# 3. Read data and select the sample
# ----------------------------------------------------------------------------
mfile = fits.open(MASTER_PATH)
phot = Table(mfile[1].data)
phys = Table(mfile[2].data)

condition_clean = np.logical_and.reduce((
    phot['warn_flag'] == 0,                       # No warning flag
    np.abs(phot['mag_auto_f444w']) < 30,          # Remove very faint objects
    np.abs(phot['mag_auto_f277w']) < 30,
    phot['flag_star_hsc'] == 0,                   # Remove objects in HSC star mask area
    phot['flux_auto_f277w'] > 0,                  # Remove objects with negative flux
    phot['flux_auto_f444w'] > 0,
    phys['chi_star'] > 10.0,
    np.abs(phys['zfinal'] - phys['zpdf_l68']) < 1.0,
    phot['snr_f277w'] > 5,
    phot['snr_f444w'] > 5,
))

# Parent sample: the mass cut is applied *per draw* below (as in the paper), so keep every
# source whose median mass is close enough that its posterior could exceed M_LIM.
parent = condition_clean & (phys['mass_med'] > M_LIM - 0.5) & (phys['zfinal'] > Z_EDGES[0] - 1.0)

# ERO flag: F277W - F444W > 1.5.  (The paper also requires them to be point-like in F444W;
# you can add e.g. a cut on phot['c_f444w'] here.)
color = phot['mag_auto_f277w'] - phot['mag_auto_f444w']
is_ero = np.asarray(color > COLOR_CUT)[parent]

print(f"Parent sample: {parent.sum()} sources ({is_ero.sum()} EROs)")

# Redshift / mass "posteriors" from (median, 16th, 84th percentiles)
z_med = np.asarray(phys['zfinal'][parent], float)
z_pdf = np.asarray(phys['zpdf_med'][parent], float)
z_draw = draw_split_normal(z_med,
                           z_pdf - np.asarray(phys['zpdf_l68'][parent], float),
                           np.asarray(phys['zpdf_u68'][parent], float) - z_pdf,
                           N_DRAWS, rng)

m_med = np.asarray(phys['mass_med'][parent], float)
m_draw = draw_split_normal(m_med,
                           m_med - np.asarray(phys['mass_l68'][parent], float),
                           np.asarray(phys['mass_u68'][parent], float) - m_med,
                           N_DRAWS, rng)

volumes = np.array([shell_volume(Z_EDGES[i], Z_EDGES[i + 1], SURVEY_AREA_DEG2, cosmo)
                    for i in range(len(Z_EDGES) - 1)])
z_cen = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])

# All sources (dotted) and with EROs removed (solid) -- same draws for both
all_med, all_lo, all_hi, all_N = binned_density(z_draw, m_draw, Z_EDGES, volumes, rng)
nor_med, nor_lo, nor_hi, nor_N = binned_density(z_draw[:, ~is_ero], m_draw[:, ~is_ero],
                                                Z_EDGES, volumes, rng)

# ----------------------------------------------------------------------------
# 4. Theory curves + plot
# ----------------------------------------------------------------------------
z_grid = np.linspace(Z_EDGES[0] - 0.5, Z_EDGES[-1], 40)
theory = theory_density(z_grid, EPS_LIST)

fig, ax = plt.subplots(figsize=(8, 6))

colors = plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(EPS_LIST)))
for c, eps in zip(colors, EPS_LIST):
    ax.plot(z_grid, theory[eps], color=c, lw=2, label=rf'$\epsilon = {eps}$')

def _plot_obs(med, lo, hi, ls, alpha, label):
    ok = med > 0                                   # can't show n = 0 on a log axis
    ax.plot(z_cen[ok], med[ok], 'k', ls=ls, marker='o', ms=4, label=label)
    ax.fill_between(z_cen[ok], lo[ok], hi[ok], color='k', alpha=alpha, lw=0)

_plot_obs(nor_med, nor_lo, nor_hi, '-', 0.25, 'COSMOS-Web (EROs removed)')
_plot_obs(all_med, all_lo, all_hi, ':', 0.10, 'COSMOS-Web (incl. EROs)')

ax.set(xlabel='Redshift (z)', ylabel=r'$n(>M_\star)$ [Mpc$^{-3}$, comoving]', yscale='log',
       xlim=(Z_EDGES[0] - 0.5, Z_EDGES[-1]))
ax.text(0.96, 0.95, rf'$\log\,M_\star/M_\odot > {M_LIM:g}$', transform=ax.transAxes, fontsize=12,
        ha='right', va='top')
ax.legend(fontsize=9, loc='lower left', ncol=2)
fig.tight_layout()
fig.savefig('fig5_recreation.png', dpi=200)

# ----------------------------------------------------------------------------
# 5. Efficiency needed to match each bin (cf. Fig. 10a of the paper)
# ----------------------------------------------------------------------------
print("\n z-bin      N(no ERO)   n_obs [Mpc^-3]   eps needed")
for k in range(len(z_cen)):
    eps_k = eps_needed(nor_med[k], z_cen[k]) if nor_med[k] > 0 else np.nan
    print(f" {Z_EDGES[k]:.0f}-{Z_EDGES[k+1]:.0f}     {nor_N[k]:7.1f}     {nor_med[k]:.3e}      {eps_k:.3f}")

plt.show()