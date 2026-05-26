"""DT plotting: 3-panel Data | Model | Residual figure with shared
colorbar (style matches user reference: ../20240702_Jace_XO3/main.ipynb).

Public entry points:

- :func:`plot_dt`        — pure plotting given (data, model, residual) +
                          phase array.
- :func:`make_dt_plot`   — convenience wrapper that takes the params
                          dict, computes the model via dopptom_chi2,
                          and saves a PDF.  Mirrors make_sed_plot in
                          api shape (params, datadir, outdir, outfile).
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_dt(data, model, t_axis, *,
            vel=None, vel_range=None, vel_zoom_kms=None,
            y_label='Time from mid-transit (hr)',
            y_extend_t14=True,
            label_fontsize=15, cmap='RdBu_r',
            cbar_scale=1000.0, cbar_label='Residual × 1000',
            clip_percentile=95.0,
            savepath=None, title=None,
            vsini=None, t14_hr=None, t23_hr=None,
            v_p_predicted=None, t_traversing=None):
    """Three-panel Data | Model | Residual figure with shared colorbar.

    Style follows the user's DT_NEID notebook (cell 15):
    - 98th-percentile symmetric colour clipping
    - ``interpolation='nearest'``
    - optional overlays:
        * predicted planet trajectory `v_p_predicted(t)`
        * vertical reference lines at ±vsini and v=0
        * horizontal reference lines at ±T14/2 and t=0

    Parameters
    ----------
    data, model : (nvels, ntimes) ndarray
        Observed CCF residual and model.
    t_axis : (ntimes,) ndarray
        Per-frame value for the vertical axis (default: hours from Tc).
    vel : (nvels,) ndarray, optional
        Velocity grid (km/s).
    vel_range : (vmin, vmax), optional
        x-axis limits.
    y_label : str
        Label for the y-axis.
    label_fontsize : int
    cmap : str
        Matplotlib colormap.  Default 'RdBu_r' (red = positive shadow).
    savepath : str, optional
        If provided, save to this path; otherwise return the figure.
    title : str, optional
        Suptitle.
    vsini : float, optional
        Add gray dotted lines at ±vsini (km/s).
    t14_hr : float, optional
        Add labelled lines at ±T14/2 (1st / 4th contact = ingress start /
        egress end), if they fall inside the data time range.
    t23_hr : float, optional
        Add lines at ±T23/2 (2nd / 3rd contact = ingress end / egress
        start), if covered.
    v_p_predicted : (ntimes,) ndarray, optional
        Predicted planet trajectory.  Plotted as black dashed line over
        the indices given by ``t_traversing`` (or all if None).
    t_traversing : (ntimes,) boolean array, optional
        Mask of in-transit indices for the trajectory overlay.
    """
    residual = data - model
    if vel is None:
        vel = np.arange(data.shape[0])
    # Auto velocity zoom: a few × vsini, capped by data extent
    if vel_range is None:
        if vel_zoom_kms is None:
            if vsini is not None and np.isfinite(vsini) and vsini > 0:
                vel_zoom_kms = max(15.0, 5.0 * vsini)
            else:
                vel_zoom_kms = float(np.abs(vel).max())
        vel_range = (max(float(vel.min()), -vel_zoom_kms),
                     min(float(vel.max()), +vel_zoom_kms))

    # y-axis: data range + (optionally) full transit window
    y_lo = float(t_axis.min())
    y_hi = float(t_axis.max())
    if y_extend_t14 and t14_hr is not None and np.isfinite(t14_hr) and t14_hr > 0:
        y_lo = min(y_lo, -t14_hr / 2.0 - 1.0)
        y_hi = max(y_hi, +t14_hr / 2.0 + 1.0)
    y_range = (y_lo, y_hi)

    # Symmetric clip_percentile colour range across the three panels.
    # Scale by cbar_scale (e.g. ×1000) so the colorbar uses readable numbers.
    _q = float(clip_percentile)
    _abs_max = float(max(
        np.nanpercentile(np.abs(data), _q),
        np.nanpercentile(np.abs(model), _q),
        np.nanpercentile(np.abs(residual), _q),
    ))
    if _abs_max == 0 or not np.isfinite(_abs_max):
        _abs_max = 1.0
    vmin, vmax = -_abs_max, _abs_max

    fig = plt.figure(figsize=(12, 4))
    ax_data = fig.add_axes([0.05, 0.10, 0.28, 0.80])
    ax_mod  = fig.add_axes([0.33, 0.10, 0.28, 0.80])
    ax_res  = fig.add_axes([0.61, 0.10, 0.28, 0.80])
    ax_cbar = fig.add_axes([0.92, 0.10, 0.02, 0.80])

    # imshow extent uses the *data* time range; y_range may extend further
    # to include ingress/egress markers, which we set via set_ylim below.
    data_extent = [float(vel.min()), float(vel.max()),
                   float(t_axis.min()), float(t_axis.max())]
    imshow_kw = dict(
        aspect='auto', origin='lower',
        extent=data_extent,
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest',
    )

    panel_specs = (
        (ax_data, data,     'Data'),
        (ax_mod,  model,    'Model'),
        (ax_res,  residual, 'Residual'),
    )
    for ax, arr, name in panel_specs:
        im = ax.imshow(arr.T, **imshow_kw)
        ax.set_xlabel('Velocity (km/s)', fontsize=label_fontsize)
        ax.text(0.97, 0.97, name, ha='right', va='top',
                transform=ax.transAxes, fontsize=label_fontsize,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.7, pad=2))

        # Reference lines (notebook style)
        ax.axvline(0, c='k', ls=':', lw=0.6)
        if vsini is not None and np.isfinite(vsini) and vsini > 0:
            ax.axvline(-vsini, c='gray', ls=':', lw=0.5)
            ax.axvline(+vsini, c='gray', ls=':', lw=0.5)
        # Contact-time markers — only drawn when actually inside the data
        # time window, so plots that don't cover ingress/egress stay clean.
        _y_lo, _y_hi = y_range
        _within = lambda y: _y_lo <= y <= _y_hi
        # 1st / 4th contact (T14): labelled "ingress" / "egress"
        if t14_hr is not None and np.isfinite(t14_hr) and t14_hr > 0:
            for sign, lbl in ((-1, 'ingress'), (+1, 'egress')):
                y = sign * t14_hr / 2.0
                if _within(y):
                    ax.axhline(y, c='blue', ls=':', lw=0.7)
                    if ax is ax_data:   # label only on left-most panel
                        ax.text(vel_range[0] + 0.02 * (vel_range[1] - vel_range[0]),
                                y, f' {lbl}', va='bottom', ha='left',
                                fontsize=8, color='blue')
        # 2nd / 3rd contact (T23): dashed, no label
        if t23_hr is not None and np.isfinite(t23_hr) and t23_hr > 0:
            for sign in (-1, +1):
                y = sign * t23_hr / 2.0
                if _within(y):
                    ax.axhline(y, c='blue', ls=(0, (3, 3)), lw=0.5, alpha=0.7)
        ax.axhline(0, c='k', ls='--', lw=0.6)

        # Predicted trajectory
        if v_p_predicted is not None:
            t_arr = np.asarray(t_axis)
            v_arr = np.asarray(v_p_predicted)
            if t_traversing is None:
                m = np.ones_like(t_arr, dtype=bool)
            else:
                m = np.asarray(t_traversing, dtype=bool)
            ax.plot(v_arr[m], t_arr[m], 'k--', lw=1.0, alpha=0.7)

    # Set axis ranges: velocity zoom on x, full transit window on y.
    for ax in (ax_data, ax_mod, ax_res):
        ax.set_xlim(vel_range[0], vel_range[1])
        ax.set_ylim(y_range[0], y_range[1])
    ax_data.set_ylabel(y_label, fontsize=label_fontsize)
    ax_mod.set_yticks([])
    ax_res.set_yticks([])

    cbar = fig.colorbar(im, cax=ax_cbar, orientation='vertical')
    cbar.set_label(cbar_label, fontsize=label_fontsize)
    # Show colorbar ticks scaled by cbar_scale (default ×1000)
    import matplotlib.ticker as mticker
    cbar.formatter = mticker.FuncFormatter(lambda x, _: f'{x * cbar_scale:.2f}')
    cbar.update_ticks()

    if title:
        fig.suptitle(title, fontsize=label_fontsize, y=1.02)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
        return savepath
    return fig


# ---------------------------------------------------------------------------
#  Pipeline-stage entry point (mirror of make_sed_plot)
# ---------------------------------------------------------------------------
def make_dt_plot(params, datadir, outdir, *, outfile, inst,
                 dt_data=None, basement=None):
    """Compute model + save 3-panel DT figure for one instrument.

    Used at the initial_guess / optimized / mcmc pipeline stages, mirroring
    make_sed_plot / make_mist_plot.

    Parameters
    ----------
    params : dict
        Fully-resolved parameter dict (output of update_params + extras).
    datadir : str
        Fit directory (unused here but kept for API consistency).
    outdir : str
        Where to write the PDF.
    outfile : str
        Filename within ``outdir`` (e.g. 'mcmc_dt_TRES.pdf').
    inst : str
        DT instrument label (key in basement.dt_data).
    dt_data : dict, optional
        Pre-loaded DT data (read_dt_fits output).  If None, pulled from
        ``basement.dt_data[inst]``.
    basement : Basement, optional
        Fallback when ``dt_data`` is not supplied.

    Returns
    -------
    path or None
        Full path of the saved PDF, or None if the model could not be
        computed (e.g. parameters invalid).
    """
    from .core import dopptom_chi2
    from ..utils.quadld import quadld

    if dt_data is None:
        if basement is None:
            from .. import config
            basement = config.BASEMENT
        dt_data = basement.dt_data[inst]
    if basement is None:
        from .. import config
        basement = config.BASEMENT

    companion = basement.settings['companions_phot'][0]

    # Stellar logg
    _Msun_kg = 1.989e30; _Rsun_m = 6.957e8; _G_si = 6.674e-11
    try:
        mstar = float(params['A_mstar'])
        rstar = float(params['A_rstar'])
        g_cgs = _G_si * mstar * _Msun_kg / (rstar * _Rsun_m) ** 2 * 100.0
        logg = float(np.log10(g_cgs))
    except Exception:
        return None
    teff = float(params.get('A_teff', np.nan))
    feh  = float(params.get('A_feh',  0.0))

    rr   = float(params[companion + '_rr'])
    rsuma = params.get(companion + '_rsuma')
    if rsuma is None or not np.isfinite(rsuma) or rsuma <= 0:
        return None
    ar   = (1.0 + rr) / float(rsuma)
    cosi = float(params[companion + '_cosi'])
    _fc  = float(params[companion + '_f_c'])
    _fs  = float(params[companion + '_f_s'])
    e    = _fc * _fc + _fs * _fs
    w    = float(np.mod(np.arctan2(_fs, _fc), 2 * np.pi))
    tc   = float(params[companion + '_epoch'])
    per  = float(params[companion + '_period'])
    # allesfast stores lambda in degrees; dopptom_chi2 wants radians
    _lam_deg = params.get(companion + '_lambda')
    if _lam_deg is None:
        _lam_deg = params.get('A_lambda')
    if _lam_deg is None:
        _lam_deg = 0.0
    lam = float(_lam_deg) * np.pi / 180.0

    band = basement.settings.get(f'dt_ld_band_{inst}', 'V')
    u1, u2 = quadld(logg, teff, feh, band)
    if not (np.isfinite(u1) and np.isfinite(u2)):
        return None

    vsini = float(params.get('A_vsini', np.nan))
    vline = float(params.get(f'A_vline_{inst}', np.nan))
    errs  = float(params.get(f'A_dt_errscale_{inst}', 1.0))
    if not (np.isfinite(vsini) and np.isfinite(vline)):
        return None

    chi2, model = dopptom_chi2(
        dt_data, tc, per, e, w, cosi, rr, ar, lam,
        float(u1), float(u2), vsini, vline, errs,
        return_model=True,
    )
    if model is None or not np.all(np.isfinite(model)):
        return None

    # y-axis: hours from the nearest mid-transit (Tc + N·P)
    bjd = dt_data['bjd']
    n_orbits = np.round(np.mean(bjd - tc) / per)
    tc_near = tc + n_orbits * per
    t_axis = (bjd - tc_near) * 24.0   # hours

    # Mean-subtract data so colormap centres on zero
    data_centered = dt_data['ccf2d'] - dt_data['median_ccf']
    model_centered = model - dt_data['median_ccf']

    # Predicted planet trajectory v_p(t) = vsini * x_p(t)/R*
    from .geometry import primary_transit_phase, sky_positions
    inc = np.arccos(cosi)
    tp = tc_near - per * primary_transit_phase(e, w)
    x_p, y_p, _z_p, b_proj = sky_positions(bjd, inc, ar, tp, per, e, w)
    # Rotation around lambda: LOS velocity from x_p in stellar frame
    up_norm = x_p * np.cos(lam) - y_p * np.sin(lam)   # in vsini units
    v_p_pred = vsini * up_norm
    r_sky = np.sqrt(x_p ** 2 + y_p ** 2)
    is_traversing = r_sky < (1.0 + rr)

    # Transit durations:
    #   Circular: T = (P/π)·arcsin(√[(1±k)² − b²] / (a·sin i))  (S+M-O 2003)
    #   Eccentric correction (Winn 2010 Eq 16):
    #     T_ecc = T_circ · √(1−e²) / (1 + e·sin ω)
    sini = np.sin(inc)
    b_impact_sq = (ar * cosi) ** 2
    arg14 = (1.0 + rr) ** 2 - b_impact_sq
    arg23 = (1.0 - rr) ** 2 - b_impact_sq
    ecc_factor = (np.sqrt(max(0.0, 1.0 - e ** 2))
                   / max(1e-6, 1.0 + e * np.sin(w)))
    if arg14 > 0 and sini > 0:
        t14_hr = (per / np.pi) * np.arcsin(np.sqrt(arg14) / (ar * sini)) * 24.0
        t14_hr *= ecc_factor
    else:
        t14_hr = None
    if arg23 > 0 and sini > 0:
        t23_hr = (per / np.pi) * np.arcsin(np.sqrt(arg23) / (ar * sini)) * 24.0
        t23_hr *= ecc_factor
    else:
        t23_hr = None

    path = os.path.join(outdir, outfile)
    title = dt_data.get('label', f'DT {inst}')
    plot_dt(data_centered, model_centered, t_axis,
            vel=dt_data['vel'], savepath=path, title=title,
            vsini=vsini, t14_hr=t14_hr, t23_hr=t23_hr,
            v_p_predicted=v_p_pred, t_traversing=is_traversing)
    return path
