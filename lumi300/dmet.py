def getVarDict(fb, process, var_name=None):
    var_dict = {}
    if var_name is None or var_name == 'met':
        var_dict['met'] = {
            'var': fb['met_tst_et'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ [{\rm MeV}]$'
        }
    if var_name is None or var_name == 'ph_pt':
        var_dict['ph_pt'] = {
            'var': ak.firsts(fb['ph_pt']),
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{\gamma}\;\mathrm{or}\;p_T^{e}\;(e\to\gamma)\;[{\rm MeV}]$'
        }
    if var_name is None or var_name == 'metsig':
        var_dict['metsig'] = {
            'var': fb['met_tst_sig'],
            'bins': np.linspace(0, 30, 15+1),
            'title': r'$E_T^{miss}\ \mathrm{significance}$'
        }
    if var_name is None or var_name == 'dphi_met_phterm':
        var_dict['dphi_met_phterm'] = {
            'var': fb['dphi_met_phterm'],
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})$'
        }
    return var_dict


def scale_good_bad_migration(w_good, w_bad, pct):
    """
    Preserves total yield while scaling Good PV up and Bad PV down
    to match the target Good PV purity (pct %).
    """
    P = pct / 100.0
    sum_g = np.sum(w_good)
    sum_b = np.sum(w_bad)
    tot = sum_g + sum_b

    if tot == 0:
        return w_good, w_bad

    target_g = tot * P
    target_b = tot * (1.0 - P)

    scale_g = (target_g / sum_g) if sum_g > 0 else 0.0
    scale_b = (target_b / sum_b) if sum_b > 0 else 0.0

    return w_good * scale_g, w_bad * scale_b


def plot_pv_migration_overlays(
    tot_array,
    pv_pcts=range(0, 101, 10),
    variables_to_plot=['met', 'ph_pt', 'metsig', 'dphi_met_phterm']
):
    base_out_dir = "/home/jlai/dark_photon/lumi300/plots_migration_overlay"
    os.makedirs(base_out_dir, exist_ok=True)

    fb_sig = tot_array[0]
    fb_bkg = tot_array[1]

    sig_good = ak.to_numpy(fb_sig['is_good_pv'])
    bkg_good = ak.to_numpy(fb_bkg['is_good_pv'])

    sig_w_0 = ak.to_numpy(fb_sig['weights'])
    bkg_w_0 = ak.to_numpy(fb_bkg['weights'])

    sig_w_good_0, sig_w_bad_0 = sig_w_0[sig_good], sig_w_0[~sig_good]
    bkg_w_good_0, bkg_w_bad_0 = bkg_w_0[bkg_good], bkg_w_0[~bkg_good]

    colors = {
        "bkg_good": "#4C72B0",  # Steel Blue
        "bkg_bad":  "#DD8452",  # Coral Orange
        "sig_good": "#D62728",  # Crimson Red
        "sig_bad":  "#9467BD"   # Amethyst Purple
    }

    for var in variables_to_plot:
        sig_cfg = getVarDict(fb_sig, 'ggHyyd', var_name=var)[var]
        bkg_cfg = getVarDict(fb_bkg, 'gammajet_direct', var_name=var)[var]

        bins = sig_cfg['bins']
        title = sig_cfg['title']

        sig_x = ak.to_numpy(sig_cfg['var'])
        bkg_x = ak.to_numpy(bkg_cfg['var'])

        sig_x_good, sig_x_bad = sig_x[sig_good], sig_x[~sig_good]
        bkg_x_good, bkg_x_bad = bkg_x[bkg_good], bkg_x[~bkg_good]

        for pct in pv_pcts:
            # Apply migration scaling to both Signal and Background
            sig_w_good, sig_w_bad = scale_good_bad_migration(sig_w_good_0, sig_w_bad_0, pct)
            bkg_w_good, bkg_w_bad = scale_good_bad_migration(bkg_w_good_0, bkg_w_bad_0, pct)

            components = [
                (bkg_x_good, bkg_w_good, colors["bkg_good"], "-",  rf'$\gamma$+jets (Good PV) [$\Sigma w$: {np.sum(bkg_w_good):.0f}]'),
                (bkg_x_bad,  bkg_w_bad,  colors["bkg_bad"],  "--", rf'$\gamma$+jets (Bad PV) [$\Sigma w$: {np.sum(bkg_w_bad):.0f}]'),
                (sig_x_good, sig_w_good, colors["sig_good"], "-",  rf'ggHyyd (Good PV) [$\Sigma w$: {np.sum(sig_w_good):.1f}]'),
                (sig_x_bad,  sig_w_bad,  colors["sig_bad"],  "--", rf'ggHyyd (Bad PV) [$\Sigma w$: {np.sum(sig_w_bad):.1f}]'),
            ]

            fig, ax = plt.subplots(figsize=(12, 9))

            for x_vals, w_vals, col, ls, lbl in components:
                ax.hist(
                    x_vals,
                    bins=bins,
                    weights=w_vals,
                    color=col,
                    linestyle=ls,
                    label=lbl,
                    histtype='step',
                    linewidth=2.5,
                    density=False
                )

                # Statistical error bars: sqrt(sum(w^2))
                if len(x_vals) > 0 and np.sum(w_vals) > 0:
                    counts, bin_edges = np.histogram(x_vals, bins=bins, weights=w_vals)
                    sum_w2, _ = np.histogram(x_vals, bins=bins, weights=w_vals**2)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    ax.errorbar(
                        bin_centers,
                        counts,
                        yerr=np.sqrt(sum_w2),
                        fmt='none',
                        ecolor=col,
                        elinewidth=1.5,
                        capsize=0
                    )

            ax.set_yscale('log')
            ax.set_ylim(1e-2, 1e9)
            ax.set_xlim(bins[0], bins[-1])
            ax.minorticks_on()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_xlabel(title)
            ax.set_ylabel("Weighted Events")
            ax.set_title(f"{var} | Good PV Purity (Migration): {pct}%")
            ax.legend(ncol=2, loc="upper right")

            plt.tight_layout()
            out_path = f"{base_out_dir}/{var}_migration_pv{pct:03d}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()

        print(f"Saved all migration overlay plots for: {var}")


# Run across Good PV purity targets from 0% to 100%
plot_pv_migration_overlays(tot2, pv_pcts=range(0, 101, 10))