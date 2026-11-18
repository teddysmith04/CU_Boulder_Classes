"""
sed_plot.py -- plot observed-frame SEDs from the COSMOS2025 master catalog.

Expected input
--------------
`filedata` is a sequence of two row-aligned tables:
    filedata[0] : SE++ photometry  (flux_model_*, flux_err-cal_model_*, id, ...)
    filedata[1] : LePhare outputs  (zfinal, mass_med, chi2_best, chi2_agn, ...)
Anything that supports tbl['col'][i] works (astropy Table, FITS_rec, dict of arrays).

IMPORTANT ASSUMPTIONS TO VERIFY
-------------------------------
1. Flux units. The code assumes flux_model_* is in microJy (AB zero point 23.9).
   Run check_flux_units(filedata[0]) once; it compares flux_model_* to mag_model_*
   and tells you the conversion factor to pass as `to_uJy`.
2. Filter wavelengths in BAND_INFO are approximate pivot wavelengths typed from
   memory. They are fine for plotting, but use the SVO Filter Profile Service
   for anything quantitative.
3. mass_* columns are assumed to be log10(M*/Msun), and chi2_agn / chi2_best are
   assumed to be the AGN-template and galaxy-template fits. Confirm both
   against the COSMOS2025 paper (Shuntov et al. 2025) before interpreting them.
4. The catalog stores only model *indices* (mod_star, mod_agn, ...), not the
   best-fit spectra. To overlay a model SED, pass it via `models=`
   (e.g. read from LePhare .spec output files).

Example
-------
    from sed_plot import plot_sed, plot_sed_grid, check_flux_units
    check_flux_units(filedata[0])
    plot_sed(filedata, idx=1234)
    plot_sed(filedata, obj_id=987654, yaxis='flux')
    plot_sed_grid(filedata, [10, 20, 30, 40], ncols=2, savepath='candidates.png')
"""

import numpy as np
import matplotlib.pyplot as plt

AB_ZP_UJY = 23.9  # AB zero point for fluxes in microJy

# band name (as it appears in column names) -> (approx pivot wavelength [micron], group)
BAND_INFO = {
    # HST
    'hst-f814w': (0.8045, 'HST'),
    # JWST NIRCam / MIRI
    'f115w': (1.154, 'NIRCam'),
    'f150w': (1.501, 'NIRCam'),
    'f277w': (2.776, 'NIRCam'),
    'f444w': (4.421, 'NIRCam'),
    'f770w': (7.639, 'MIRI'),
    # Spitzer IRAC
    'irac-ch1': (3.550, 'IRAC'),
    'irac-ch2': (4.493, 'IRAC'),
    'irac-ch3': (5.731, 'IRAC'),
    'irac-ch4': (7.872, 'IRAC'),
    # Ground-based broad bands
    # 'cfht-u': (0.371, 'Ground (broad)'),
    # 'hsc-g': (0.481, 'Ground (broad)'),
    # 'hsc-r': (0.623, 'Ground (broad)'),
    # 'hsc-i': (0.774, 'Ground (broad)'),
    # 'hsc-z': (0.891, 'Ground (broad)'),
    # 'hsc-y': (0.978, 'Ground (broad)'),
    # 'uvista-y': (1.021, 'Ground (broad)'),
    # 'uvista-j': (1.252, 'Ground (broad)'),
    # 'uvista-h': (1.646, 'Ground (broad)'),
    # 'uvista-ks': (2.156, 'Ground (broad)'),
    # # Ground-based medium / narrow bands
    # 'hsc-nb0816': (0.817, 'Ground (medium/narrow)'),
    # 'hsc-nb0921': (0.921, 'Ground (medium/narrow)'),
    # 'hsc-nb1010': (1.010, 'Ground (medium/narrow)'),
    # 'uvista-nb118': (1.191, 'Ground (medium/narrow)'),
    # 'sc-ib427': (0.426, 'Ground (medium/narrow)'),
    # 'sc-ia484': (0.485, 'Ground (medium/narrow)'),
    # 'sc-ib505': (0.506, 'Ground (medium/narrow)'),
    # 'sc-ia527': (0.526, 'Ground (medium/narrow)'),
    # 'sc-ib574': (0.576, 'Ground (medium/narrow)'),
    # 'sc-ia624': (0.623, 'Ground (medium/narrow)'),
    # 'sc-ia679': (0.678, 'Ground (medium/narrow)'),
    # 'sc-ib709': (0.707, 'Ground (medium/narrow)'),
    # 'sc-nb711': (0.712, 'Ground (medium/narrow)'),
    # 'sc-ia738': (0.736, 'Ground (medium/narrow)'),
    # 'sc-ia767': (0.769, 'Ground (medium/narrow)'),
    # 'sc-nb816': (0.815, 'Ground (medium/narrow)'),
    # 'sc-ib827': (0.824, 'Ground (medium/narrow)'),
}

GROUP_STYLE = {  # color, marker, size
    'HST': ('tab:purple', 'D', 6),
    'NIRCam': ('tab:blue', 's', 7),
    'MIRI': ('tab:red', '*', 12),
    'IRAC': ('tab:orange', '^', 7),
    'Ground (broad)': ('0.35', 'o', 6),
    'Ground (medium/narrow)': ('0.7', 'o', 5),
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _val(tbl, col, i):
    """Float value of tbl[col][i]; NaN if the column is missing or not numeric."""
    try:
        v = tbl[col][i]
    except (KeyError, ValueError, IndexError):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _clean(v):
    """NaN out non-finite values and the usual -99 sentinels."""
    return v if (np.isfinite(v) and v > -98) else np.nan


def _fmt(v, spec='.2f'):
    v = _clean(v)
    return '--' if not np.isfinite(v) else format(v, spec)


def _resolve_index(phot, idx, obj_id):
    if (idx is None) == (obj_id is None):
        raise ValueError("Give exactly one of `idx` or `obj_id`.")
    if idx is not None:
        return int(idx)
    hits = np.where(np.asarray(phot['id']) == obj_id)[0]
    if hits.size == 0:
        raise ValueError(f"id {obj_id} not found in photometry table.")
    return int(hits[0])


def check_flux_units(phot, band='f444w', nmax=5000):
    """
    Compare flux_model_<band> with mag_model_<band> to infer the flux zero point.
    Prints the implied ZP and the `to_uJy` factor to pass to plot_sed.
    (ZP = 23.9 means the fluxes are already in microJy -> to_uJy = 1.)
    """
    f = np.asarray(phot[f'flux_model_{band}'], dtype=float)[:nmax]
    m = np.asarray(phot[f'mag_model_{band}'], dtype=float)[:nmax]
    ok = np.isfinite(f) & np.isfinite(m) & (f > 0) & (m > 0) & (m < 50)
    if ok.sum() == 0:
        print("No usable rows to infer the flux zero point.")
        return None
    zp = np.median(m[ok] + 2.5 * np.log10(f[ok]))
    to_uJy = 10 ** (0.4 * (AB_ZP_UJY - zp))
    print(f"Implied zero point from {band}: {zp:.3f} (AB, uJy -> 23.900)")
    print(f"=> use to_uJy = {to_uJy:.4g}")
    return to_uJy


def _photometry(phot, i, bands, err_kind, to_uJy, snr_min, nsigma_ul):
    """Return dict group -> {'det': [(lam, f, e)], 'ul': [(lam, f_lim)]} (fluxes in uJy)."""
    out = {}
    for b in bands:
        lam, grp = BAND_INFO[b]
        f = _clean(_val(phot, f'flux_model_{b}', i)) * to_uJy
        e = _clean(_val(phot, f'flux_err-{err_kind}_model_{b}', i)) * to_uJy
        # np.nan * to_uJy stays nan; skip bands that are not covered / not measured
        if not (np.isfinite(f) and np.isfinite(e)) or e <= 0:
            # flux can legitimately be <= 0 (undetected), but we need a valid error
            if not (np.isfinite(e) and e > 0):
                continue
            f = 0.0
        d = out.setdefault(grp, {'det': [], 'ul': []})
        if f / e >= snr_min:
            d['det'].append((lam, f, e))
        else:
            d['ul'].append((lam, nsigma_ul * e))
    return out


def _to_y(flux_uJy, yaxis):
    flux_uJy = np.asarray(flux_uJy, dtype=float)
    if yaxis == 'flux':
        return flux_uJy
    return AB_ZP_UJY - 2.5 * np.log10(flux_uJy)


# ----------------------------------------------------------------------------
# main plotting function
# ----------------------------------------------------------------------------
def plot_sed(filedata, idx=None, *, obj_id=None, ax=None, yaxis='mag',
             snr_min=3.0, nsigma_ul=2.0, err_kind='cal', to_uJy=1.0,
             z_col='zfinal', bands=None, models=None, annotate=True,
             rest_axis=True, legend=True, fontsize=9, savepath=None):
    """
    Plot the observed-frame SED of one COSMOS2025 source.

    Parameters
    ----------
    filedata : [phot_table, lephare_table], row-aligned.
    idx / obj_id : row index, or value of the `id` column (give exactly one).
    ax : existing Axes to draw on (else a new figure is made).
    yaxis : 'mag' (AB, inverted axis) or 'flux' (f_nu in microJy, log scale).
    snr_min : bands with flux/err below this are drawn as upper limits.
    nsigma_ul : upper limits are drawn at nsigma_ul * err.
    err_kind : 'cal' -> flux_err-cal_model_*, 'uncal' -> flux_err-uncal_model_*.
    to_uJy : multiply catalog fluxes by this to get microJy (see check_flux_units).
    z_col : redshift column in the LePhare table used for the rest-frame axis.
    bands : subset of BAND_INFO keys to plot (default: all available).
    models : optional list of (wave_um, flux_uJy, label) model SEDs to overlay.
    savepath : if given, save the figure there.

    Returns
    -------
    (fig, ax)
    """
    if yaxis not in ('mag', 'flux'):
        raise ValueError("yaxis must be 'mag' or 'flux'.")
    phot, lep = filedata[0], filedata[1]
    i = _resolve_index(phot, idx, obj_id)
    bands = list(bands) if bands is not None else list(BAND_INFO)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    data = _photometry(phot, i, bands, err_kind, to_uJy, snr_min, nsigma_ul)
    yvals = []

    for grp in GROUP_STYLE:  # fixed legend order
        if grp not in data:
            continue
        color, marker, ms = GROUP_STYLE[grp]
        det, ul = data[grp]['det'], data[grp]['ul']
        if det:
            lam, f, e = np.array(det).T
            y = _to_y(f, yaxis)
            if yaxis == 'mag':
                yerr = 1.0857 * e / f
            else:  # asymmetric so the lower bar never crosses zero on a log axis
                yerr = np.vstack([np.minimum(e, 0.95 * f), e])
            ax.errorbar(lam, y, yerr=yerr, fmt=marker, color=color, ms=ms,
                        mec='k', mew=0.6, ecolor=color, elinewidth=1.2,
                        capsize=2, label=grp, zorder=3)
            yvals += list(y)
        if ul:
            lam, flim = np.array(ul).T
            y = _to_y(flim, yaxis)
            # 'v' points down on screen = lower flux (flux axis) / fainter (inverted mag axis)
            ax.scatter(lam, y, marker='v', s=40, facecolors='none',
                       edgecolors=color, linewidths=1.2, zorder=2,
                       label=None if det else grp)
            yvals += list(y)

    if models:
        for wave, flux, label in models:
            ax.plot(wave, _to_y(flux, yaxis), lw=1, alpha=0.8, label=label, zorder=1)

    # axes cosmetics
    ax.set_xscale('log')
    ax.set_xlim(0.3, 9.5)
    ax.set_xticks([0.4, 0.6, 1, 2, 3, 5, 8])
    ax.set_xticklabels(['0.4', '0.6', '1', '2', '3', '5', '8'])
    ax.minorticks_on()
    ax.set_xlabel(r'Observed wavelength [$\mu$m]')

    if yaxis == 'mag':
        ax.set_ylabel('AB magnitude')
        if yvals:
            ax.set_ylim(max(yvals) + 0.7, min(yvals) - 0.7)  # inverted
    else:
        ax.set_yscale('log')
        ax.set_ylabel(r'$f_\nu$ [$\mu$Jy]')
        if yvals:
            ax.set_ylim(min(yvals) / 2.0, max(yvals) * 2.0)

    # secondary rest-frame wavelength axis
    z = _clean(_val(lep, z_col, i))
    if rest_axis and np.isfinite(z) and z >= 0:
        sec = ax.secondary_xaxis('top', functions=(lambda x: x / (1 + z),
                                                   lambda x: x * (1 + z)))
        sec.set_xlabel(r'Rest-frame wavelength [$\mu$m]')

    oid = phot['id'][i] if 'id' in getattr(phot, 'columns', phot) else i
    ax.set_title(f'ID {oid}', pad=34 if rest_axis else 6, fontsize=fontsize + 2)

    if annotate:
        lines = _annotation_lines(phot, lep, i, z_col, to_uJy)
        ax.text(0.02, 0.5, '\n'.join(lines), transform=ax.transAxes,
                va='bottom' if yaxis == 'mag' else 'bottom', ha='left',
                fontsize=fontsize, family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))

    if legend:
        ax.legend(loc='upper right', fontsize=fontsize - 1, framealpha=0.9)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close(fig)
    return fig, ax


def _annotation_lines(phot, lep, i, z_col, to_uJy):
    z = _val(lep, z_col, i)
    zlo, zhi = _val(lep, 'zpdf_l68', i), _val(lep, 'zpdf_u68', i)
    m, mlo, mhi = (_val(lep, c, i) for c in ('mass_med', 'mass_l68', 'mass_u68'))
    chi_g, chi_a = _val(lep, 'chi2_best', i), _val(lep, 'chi2_agn', i)
    z_agn = _val(lep, 'zp_agn', i)
    typ = _val(lep, 'type', i)

    # F277W - F444W color from the plotted fluxes
    f277, f444 = _val(phot, 'flux_model_f277w', i), _val(phot, 'flux_model_f444w', i)
    if _clean(f277) > 0 and _clean(f444) > 0:
        color = -2.5 * np.log10(f277 / f444)
    else:
        color = np.nan

    d_chi = (chi_g - chi_a) if (np.isfinite(_clean(chi_g)) and np.isfinite(_clean(chi_a))) else np.nan
    return [
        f"{z_col:<8} = {_fmt(z, '.3f')}  [{_fmt(zlo, '.2f')}, {_fmt(zhi, '.2f')}]",
        f"logM*    = {_fmt(m)}  [{_fmt(mlo)}, {_fmt(mhi)}]",
        f"chi2_gal = {_fmt(chi_g, '.1f')}",
        f"chi2_agn = {_fmt(chi_a, '.1f')}  (z_agn={_fmt(z_agn, '.2f')})",
        f"d_chi2   = {_fmt(d_chi, '.1f')}  (gal - agn)",
        f"F277W-F444W = {_fmt(color)}",
        f"type     = {_fmt(typ, '.0f')}",
    ]


# ----------------------------------------------------------------------------
# multi-object convenience wrapper
# ----------------------------------------------------------------------------
def plot_sed_grid(filedata, idxs, ncols=2, panel_size=(6.0, 4.6), savepath=None, **kwargs):
    """Plot several SEDs in a grid. `idxs` are row indices; kwargs go to plot_sed."""
    idxs = list(idxs)
    nrows = int(np.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    kwargs.setdefault('fontsize', 8)
    for k, (ax, i) in enumerate(zip(axes.ravel(), idxs)):
        plot_sed(filedata, idx=i, ax=ax, legend=(k == 0), **kwargs)
    for ax in axes.ravel()[len(idxs):]:
        ax.set_visible(False)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    return fig, axes