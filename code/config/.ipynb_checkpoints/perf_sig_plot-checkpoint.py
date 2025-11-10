def plot_performance(tot2, ntuple_names, sample_dict, getVarDict, mt_val_dir, cut_name, signal_name='ggHyyd'):
    var_config = getVarDict(tot2[0], 'ggHyyd')
    
    for var in var_config:
        bg_values = []     
        bg_weights = []    
        bg_colors = []     
        bg_labels = []     
    
        signal_values = [] 
        signal_weights = []
        signal_color = None 
        signal_label = None
    
        for j in range(len(ntuple_names)):
            process = ntuple_names[j]
            fb = tot2[j]  # TTree
            var_config = getVarDict(fb, process, var_name=var)
    
            x = var_config[var]['var'] # TBranch
            bins = var_config[var]['bins'] 
            weights = fb['weights']
            
            sample_info = sample_dict[process]
            color = sample_info['color']
            legend = sample_info['legend']
    
            
            if process == 'ggHyyd':  # signal
                signal_values.append(x)
                signal_weights.append(weights)
                signal_color = color
                signal_label = legend
            else:   # background
                bg_values.append(x)
                bg_weights.append(weights)
                bg_colors.append(color)
                bg_labels.append(legend)
    
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 13), gridspec_kw={'height_ratios': [9, 4]})
    
        ax_top.hist(bg_values, bins=bins, weights=bg_weights, color=bg_colors,
                    label=bg_labels, stacked=True)
    
        ax_top.hist(signal_values, bins=bins, weights=signal_weights, color=signal_color,
                    label=signal_label, histtype='step', linewidth=2)
    
        signal_all = np.concatenate(signal_values) if len(signal_values) > 0 else np.array([])
        signal_weights_all = np.concatenate(signal_weights) if len(signal_weights) > 0 else np.array([])
    
        # Add error bar for signal (top plot)
        if len(signal_all) > 0:
            signal_counts, bin_edges = np.histogram(signal_all, bins=bins, weights=signal_weights_all)
            sum_weights_sq, _ = np.histogram(signal_all, bins=bins, weights=signal_weights_all**2)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            signal_errors = np.sqrt(sum_weights_sq)  # Poisson error sqrt(N)
    
            ax_top.errorbar(bin_centers, signal_counts, yerr=signal_errors, fmt='.', linewidth=2,
                            color=signal_color, capsize=0)
    
        ax_top.set_yscale('log')
        ax_top.set_ylim(0.0001, 1e11)
        ax_top.set_xlim(bins[0], bins[-1])
        ax_top.minorticks_on()
        ax_top.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_top.set_ylabel("Events")
        ax_top.legend(ncol=2)
    
        bg_all = np.concatenate(bg_values) if len(bg_values) > 0 else np.array([])
        bg_weights_all = np.concatenate(bg_weights) if len(bg_weights) > 0 else np.array([])
    
        # Compute the weighted histogram counts using np.histogram
        S_counts, _ = np.histogram(signal_all, bins=bins, weights=signal_weights_all)
        B_counts, _ = np.histogram(bg_all, bins=bins, weights=bg_weights_all)     
    
        # Compute per-bin significance
        sig_simple = np.zeros_like(S_counts, dtype=float)
        sig_s_plus_b = np.zeros_like(S_counts, dtype=float)
        sig_s_plus_1p3b = np.zeros_like(S_counts, dtype=float)
    
        sqrt_B = np.sqrt(B_counts)
        sqrt_SplusB = np.sqrt(S_counts + B_counts)
        sqrt_Splus1p3B = np.sqrt(S_counts + 1.3 * B_counts)
    
        # Avoid division by zero safely
        sig_simple = np.where(B_counts > 0, S_counts / sqrt_B, 0)
        sig_s_plus_b = np.where((S_counts + B_counts) > 0, S_counts / sqrt_SplusB, 0)
        sig_s_plus_1p3b = np.where((S_counts + 1.3 * B_counts) > 0, S_counts / sqrt_Splus1p3B, 0)
    
        # Add Binomial ExpZ per bin
        zbi_per_bin = np.array([
            zbi(S_counts[i], B_counts[i], sigma_b_frac=0.3)
            for i in range(len(S_counts))
        ])
    
        # Compute the bin centers for plotting
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
        # Compute the total significance: total S / sqrt(total B)
        total_signal = np.sum(S_counts)
        total_bkg = np.sum(B_counts)
    
        if total_bkg > 0:
            total_sig_simple = total_signal / np.sqrt(total_bkg)
            total_sig_s_plus_b = total_signal / np.sqrt(total_signal + total_bkg)
            total_sig_s_plus_1p3b = total_signal / np.sqrt(total_signal + 1.3 * total_bkg)
            total_sig_binomial = zbi(total_signal, total_bkg, sigma_b_frac=0.3)
        else:
            total_sig_simple = total_sig_s_plus_b = total_sig_s_plus_1p3b = total_sig_binomial = 0
    
        # --- Plot all significance curves ---
        ax_bot.step(bin_centers, sig_simple, where='mid', color='chocolate', linewidth=2,
                    label=f"S/√B = {total_sig_simple:.4f}")
        ax_bot.step(bin_centers, sig_s_plus_b, where='mid', color='tomato', linewidth=2,
                    label=f"S/√(S+B) = {total_sig_s_plus_b:.4f}")
        ax_bot.step(bin_centers, sig_s_plus_1p3b, where='mid', color='orange', linewidth=2,
                    label=f"S/√(S+1.3B) = {total_sig_s_plus_1p3b:.4f}")
        ax_bot.step(bin_centers, zbi_per_bin, where='mid', color='plum', linewidth=2,
                    label=f"Binomial ExpZ = {total_sig_binomial:.4f}")
    
        ax_bot.set_xlabel(var_config[var]['title'])
        # ax_bot.set_xticks(np.linspace(bins[0], bins[-1], 11))
        ax_bot.set_ylabel("Significance")
        # ax_bot.set_ylim(-0.8, 2)
        ax_top.set_xlim(bins[0], bins[-1])
    
        # Do not set a title on the bottom plot.
        ax_bot.set_title("")
    
        # Draw a legend with purple text.
        leg = ax_bot.legend()
        for text in leg.get_texts():
            text.set_color('purple')
    
        plt.xlim(bins[0], bins[-1])
        plt.tight_layout()
        plt.savefig(f"/home/jlai/dark_photon/main/{mt_val_dir}/{cut_name}cut/{var}.png")
        print(f"successfully saved to /home/jlai/dark_photon/main/{mt_val_dir}/{cut_name}cut/{var}.png")
        plt.close()
        # plt.show()
    
        y_true = np.concatenate([np.ones_like(signal_all), np.zeros_like(bg_all)])
        y_scores = np.concatenate([signal_all, bg_all])
        # Combine the weights for all events.
        y_weights = np.concatenate([signal_weights_all, bg_weights_all])
    
        # Compute the weighted ROC curve.
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, sample_weight=y_weights)
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]
    
        roc_auc = auc(fpr_sorted, tpr_sorted)
    
        # Create a new figure for the ROC curve.
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, lw=2, color='red', label=f'ROC curve (AUC = {roc_auc:.5f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random chance')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {var}")
        plt.legend(loc="lower right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()    
        plt.savefig(f"/home/jlai/dark_photon/main/{mt_val_dir}/{cut_name}cut/roc_curve_{var}.png")
        plt.close()
        # plt.show()


def plot_significance(tot2, ntuple_names, sample_dict, getVarDict, cut_config, sel, mt_val_dir, cut_name, signal_name='ggHyyd'):

    def calculate_significance(cut_var, cut_type, cut_values):
        sig_simple_list = []
        sig_s_plus_b_list = []
        sig_s_plus_1p3b_list = []
        sig_binomial_list = []
    
        sigacc_simple_list = []
        sigacc_s_plus_b_list = []
        sigacc_s_plus_1p3b_list = []
        sigacc_binomial_list = []
    
        acceptance_values = []  # Store acceptance percentages
    
        for cut in cut_values:
            sig_after_cut = 0
            bkg_after_cut = []
            sig_events = 0
            
            for i in range(len(ntuple_names)):
                fb = tot2[i]
                process = ntuple_names[i]
                var_config = getVarDict(fb, process, var_name=cut_var)
                x = var_config[cut_var]['var']
                mask = x != -999 # Apply cut: Remove -999 values 
                x = x[mask]
    
                if process == signal_name:
                    sig_events = fb['weights']
                    sig_events = sig_events[mask]
                    if cut_type == 'lowercut':
                        mask = x >= cut
                    elif cut_type == 'uppercut':
                        mask = x <= cut
                    else:
                        raise ValueError("Invalid cut type")
                    sig_after_cut = ak.sum(sig_events[mask])
                
                else:
                    bkg_events = fb['weights']
                    bkg_events = bkg_events[mask]
                    if cut_type == 'lowercut':
                        mask = x >= cut
                    elif cut_type == 'uppercut':
                        mask = x <= cut
                    else:
                        raise ValueError("Invalid cut type")
                    bkg_after_cut.append(ak.sum(bkg_events[mask]))
    
           # Now compute different types of significance
            total_bkg = sum(bkg_after_cut)
            total_signal = sig_after_cut
    
            # Avoid zero division carefully
            if total_bkg > 0:
                sig_simple = total_signal / np.sqrt(total_bkg)
                sig_s_plus_b = total_signal / np.sqrt(total_signal + total_bkg) if (total_signal + total_bkg) > 0 else 0
                sig_s_plus_1p3b = total_signal / np.sqrt(total_signal + 1.3 * total_bkg) if (total_signal + 1.3*total_bkg) > 0 else 0
                sig_binomial = zbi(total_signal, total_bkg, sigma_b_frac=0.3)
            else:
                sig_simple = sig_s_plus_b = sig_s_plus_1p3b = sig_binomial = 0
    
            # Acceptance
            acceptance = total_signal / sum(sig_events) if sum(sig_events) > 0 else 0
            acceptance_values.append(acceptance * 100)  # percentage
    
            # Save significance
            sig_simple_list.append(sig_simple)
            sig_s_plus_b_list.append(sig_s_plus_b)
            sig_s_plus_1p3b_list.append(sig_s_plus_1p3b)
            sig_binomial_list.append(sig_binomial)
    
            # Save significance × acceptance
            sigacc_simple_list.append(sig_simple * acceptance)
            sigacc_s_plus_b_list.append(sig_s_plus_b * acceptance)
            sigacc_s_plus_1p3b_list.append(sig_s_plus_1p3b * acceptance)
            sigacc_binomial_list.append(sig_binomial * acceptance)
    
        return (sig_simple_list, sig_s_plus_b_list, sig_s_plus_1p3b_list, sig_binomial_list,
                sigacc_simple_list, sigacc_s_plus_b_list, sigacc_s_plus_1p3b_list, sigacc_binomial_list,
                acceptance_values)

    # Compute significance for each variable dynamically
    for cut_var, cut_types in cut_config.items():
        for cut_type, cut_values in cut_types.items():
            (sig_simple_list, sig_s_plus_b_list, sig_s_plus_1p3b_list, sig_binomial_list,
             sigacc_simple_list, sigacc_s_plus_b_list, sigacc_s_plus_1p3b_list, sigacc_binomial_list,
             acceptance_values) = calculate_significance(cut_var, cut_type, cut_values)
    
            # Plot results
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
            # Top plot: Significance vs. Cut
            ax_top.plot(cut_values, sig_simple_list, marker='o', label='S/√B')
            # ax_top.plot(cut_values, sig_s_plus_b_list, marker='s', label='S/√(S+B)')
            # ax_top.plot(cut_values, sig_s_plus_1p3b_list, marker='^', label='S/√(S+1.3B)')
            # ax_top.plot(cut_values, sig_binomial_list, marker='x', label='BinomialExpZ')
            ax_top.set_ylabel('Significance')
            ax_top.set_title(f'Significance vs. {cut_var} ({cut_type})')
            ax_top.legend()
            ax_top.grid(True)
    
            # Bottom plot: Significance * Acceptance vs. Cut
            ax_bot.plot(cut_values, sigacc_simple_list, marker='o', label='(S/√B) × Acceptance')
            # ax_bot.plot(cut_values, sigacc_s_plus_b_list, marker='s', label='(S/√(S+B)) × Acceptance')
            # ax_bot.plot(cut_values, sigacc_s_plus_1p3b_list, marker='^', label='(S/√(S+1.3B)) × Acceptance')
            # ax_bot.plot(cut_values, sigacc_binomial_list, marker='x', label='BinomialExpZ × Acceptance')
    
            for i, txt in enumerate(acceptance_values):
                ax_bot.text(cut_values[i], sigacc_simple_list[i], f'{txt:.1f}%', 
                            fontsize=10, ha='right', va='bottom', color='purple')
                
            ax_bot.set_xlabel(f'{cut_var} Cut')
            ax_bot.set_ylabel('Significance × Acceptance')
            ax_bot.set_title(f'Significance × Acceptance vs. {cut_var} ({cut_type})')
            
            ax_bot.set_xticks(cut_values)
            ax_bot.set_xticklabels(ax_bot.get_xticks(), rotation=45, ha='right')
            ax_bot.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax_bot.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))  # Show at most 10 x-ticks
            
            var_configs_tmp = getVarDict(tot2[0], signal_name, cut_var)
            ax_bot.set_xlabel(var_configs_tmp[cut_var]['title'])
            ax_bot.legend()
            ax_bot.grid(True)
    
            plt.tight_layout()
            plt.savefig(f"/home/jlai/dark_photon/main/{mt_val_dir}/{cut_name}cut/significance_{cut_var}_{cut_type}.png")
            print(f"Successfully saved to /home/jlai/dark_photon/main/{mt_val_dir}/{cut_name}cut/significance_{cut_var}_{cut_type}.png")
            plt.close()
        
    
    # --- config ---
mt_val_dir = 'mt100_140'
n_1_config = ["VertexBDTScore", "metsig", "ph_eta", "dmet", "dphi_jj", "dphi_met_jetterm"]
signal_name = 'ggHyyd'
cut_name = 'n-1'  # used to route outputs into the /{cut_name}cut/ dir

# --- helpers ---

from pathlib import Path

def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)

def to_np(a):
    """Flatten awkward/numpy to 1D numpy array; empty-safe."""
    if a is None:
        return np.array([])
    try:
        import awkward as ak
        if hasattr(ak, "to_numpy"):
            return ak.to_numpy(ak.flatten(a, axis=None))
    except Exception:
        pass
    return np.asarray(a).ravel()

def safe_concat(list_of_arrays):
    """Concatenate list of arrays safely (possibly empty)."""
    if len(list_of_arrays) == 0:
        return np.array([])
    arrs = [to_np(x) for x in list_of_arrays if x is not None]
    return np.concatenate(arrs) if len(arrs) else np.array([])

def safe_hist(x, bins, w=None):
    if x.size == 0:
        return np.zeros(len(bins)-1, dtype=float), bins
    return np.histogram(x, bins=bins, weights=w)

def sel(tot, n_1_name=None):
    """
    Apply baseline cuts to all fb in tot except the variable named by n_1_name.
    """
    import awkward as ak
    out = []
    for i in range(len(tot)):
        fb2 = tot[i]

        # VertexBDTScore > 0.12 (unless N-1 on it)
        if n_1_name != "VertexBDTScore":
            fb2 = fb2[fb2['VertexBDTScore'] >= 0.12]

        # metsig: 7 <= met_tst_sig <= 16 (only lower applied previously; fix & allow N-1)
        if n_1_name != "metsig":
            metsig_tmp = fb2['met_tst_sig']
            fb2 = fb2[(metsig_tmp >= 7) & (metsig_tmp <= 16)]

        # photon |eta| < 1.75
        if n_1_name != "ph_eta":
            ph_eta_tmp = np.abs(ak.firsts(fb2['ph_eta']))
            fb2 = fb2[ph_eta_tmp <= 1.75]

        # dmet in [-15000, 50000]
        if n_1_name != "dmet":
            dmet_tmp = fb2['dmet']
            fb2 = fb2[(dmet_tmp >= -15000) & (dmet_tmp <= 50000)]

        # dphi_met_jetterm < 0.8
        if n_1_name != "dphi_met_jetterm":
            dphi_met_jetterm_tmp = fb2['dphi_met_jetterm']
            fb2 = fb2[dphi_met_jetterm_tmp <= 0.8]

        # dphi_jj < 2.3   (fix fb->fb2, keep wrap into [0,pi], treat sentinel -10)
        if n_1_name != "dphi_jj":
            dphi_jj_tmp = fb2['dphi_central_jj']
            dphi_jj_tmp = ak.where(dphi_jj_tmp == -10, np.nan, dphi_jj_tmp)
            dphi_jj_tmp = np.arccos(np.cos(dphi_jj_tmp))
            dphi_jj_tmp = ak.where(np.isnan(dphi_jj_tmp), -999, dphi_jj_tmp)
            fb2 = fb2[dphi_jj_tmp <= 2.3]

        out.append(fb2)
    return out

def getCutDict(n_1_name=None):
    cut_dict = {}
    if n_1_name is None or n_1_name == "VertexBDTScore":
        cut_dict['VertexBDTScore'] = {'lowercut': np.arange(0.10, 0.24, 0.02)}
    if n_1_name is None or n_1_name == "dmet":
        cut_dict['dmet'] = {'lowercut': np.arange(-30000, 10000 + 5000, 5000)}
    if n_1_name is None or n_1_name == "metsig":
        cut_dict['metsig'] = {'lowercut': np.arange(0, 10 + 1, 1)}
    if n_1_name is None or n_1_name == "dphi_met_phterm":
        cut_dict['dphi_met_phterm'] = {'lowercut': np.arange(1, 2 + 0.05, 0.05)}
    if n_1_name is None or n_1_name == "dphi_met_jetterm":
        cut_dict['dphi_met_jetterm'] = {'uppercut': np.arange(0.5, 1.00, 0.05)}
    if n_1_name is None or n_1_name == "ph_eta":
        cut_dict['ph_eta'] = {'uppercut': np.arange(1.0, 2.50 + 0.05, 0.05)}
    if n_1_name is None or n_1_name == "dphi_jj":
        cut_dict['dphi_jj'] = {'uppercut': np.arange(1.0, 3.10 + 0.05, 0.05)}
    return cut_dict

# You already have calculate_significance(cut_var, cut_type, cut_values)
# assuming it uses global ntuple_names, sample_dict, tot2, signal_name, etc.

# ---------- N-1 scan + plots ----------

out_base = f"/home/jlai/dark_photon/main/{mt_val_dir}/{cut_name}cut"
ensure_dir(out_base)

def calculate_significance(cut_var, cut_type, cut_values):
    sig_simple_list = []
    sig_s_plus_b_list = []
    sig_s_plus_1p3b_list = []
    sig_binomial_list = []

    sigacc_simple_list = []
    sigacc_s_plus_b_list = []
    sigacc_s_plus_1p3b_list = []
    sigacc_binomial_list = []

    acceptance_values = []  # Store acceptance percentages

    for cut in cut_values:
        sig_after_cut = 0
        bkg_after_cut = []
        sig_events = 0
        
        for i in range(len(ntuple_names)):
            fb = tot2[i]
            process = ntuple_names[i]
            var_config = getVarDict(fb, process, var_name=cut_var)
            x = var_config[cut_var]['var']
            mask = x != -999 # Apply cut: Remove -999 values 
            x = x[mask]

            if process == signal_name:
                sig_events = fb['weights']
                sig_events = sig_events[mask]
                if cut_type == 'lowercut':
                    mask = x >= cut
                elif cut_type == 'uppercut':
                    mask = x <= cut
                else:
                    raise ValueError("Invalid cut type")
                sig_after_cut = ak.sum(sig_events[mask])
            
            else:
                bkg_events = fb['weights']
                bkg_events = bkg_events[mask]
                if cut_type == 'lowercut':
                    mask = x >= cut
                elif cut_type == 'uppercut':
                    mask = x <= cut
                else:
                    raise ValueError("Invalid cut type")
                bkg_after_cut.append(ak.sum(bkg_events[mask]))

       # Now compute different types of significance
        total_bkg = sum(bkg_after_cut)
        total_signal = sig_after_cut

        # Avoid zero division carefully
        if total_bkg > 0:
            sig_simple = total_signal / np.sqrt(total_bkg)
            sig_s_plus_b = total_signal / np.sqrt(total_signal + total_bkg) if (total_signal + total_bkg) > 0 else 0
            sig_s_plus_1p3b = total_signal / np.sqrt(total_signal + 1.3 * total_bkg) if (total_signal + 1.3*total_bkg) > 0 else 0
            sig_binomial = zbi(total_signal, total_bkg, sigma_b_frac=0.3)
        else:
            sig_simple = sig_s_plus_b = sig_s_plus_1p3b = sig_binomial = 0

        # Acceptance
        acceptance = total_signal / sum(sig_events) if sum(sig_events) > 0 else 0
        acceptance_values.append(acceptance * 100)  # percentage

        # Save significance
        sig_simple_list.append(sig_simple)
        sig_s_plus_b_list.append(sig_s_plus_b)
        sig_s_plus_1p3b_list.append(sig_s_plus_1p3b)
        sig_binomial_list.append(sig_binomial)

        # Save significance × acceptance
        sigacc_simple_list.append(sig_simple * acceptance)
        sigacc_s_plus_b_list.append(sig_s_plus_b * acceptance)
        sigacc_s_plus_1p3b_list.append(sig_s_plus_1p3b * acceptance)
        sigacc_binomial_list.append(sig_binomial * acceptance)

    return (sig_simple_list, sig_s_plus_b_list, sig_s_plus_1p3b_list, sig_binomial_list,
            sigacc_simple_list, sigacc_s_plus_b_list, sigacc_s_plus_1p3b_list, sigacc_binomial_list,
            acceptance_values)


for cut_var_tmp in n_1_config:
    # Build cut grid for THIS n-1 variable and apply baseline to all others
    cut_config = getCutDict(n_1_name=cut_var_tmp)
    tot2 = sel(tot, n_1_name=cut_var_tmp)

    # --- Significance vs cut scans for this n-1 variable ---
    for cut_var, cut_types in cut_config.items():
        for cut_type, cut_values in cut_types.items():
            (sig_simple_list, sig_s_plus_b_list, sig_s_plus_1p3b_list, sig_binomial_list,
             sigacc_simple_list, sigacc_s_plus_b_list, sigacc_s_plus_1p3b_list, sigacc_binomial_list,
             acceptance_values) = calculate_significance(cut_var, cut_type, cut_values)

            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

            # Top: S/sqrt(B) and vertical line at max
            i_max = int(np.argmax(sig_simple_list)) if len(sig_simple_list) else 0
            max_tmp = float(cut_values[i_max]) if len(cut_values) else np.nan
            if len(cut_values):
                ax_top.axvline(x=max_tmp, color='r', linestyle='--', label=f'Max S/√B at {max_tmp:.2f}')
            ax_top.plot(cut_values, sig_simple_list, marker='o', label='S/√B')
            # Uncomment if you want the other metrics on same plot:
            # ax_top.plot(cut_values, sig_s_plus_b_list, marker='s', label='S/√(S+B)')
            # ax_top.plot(cut_values, sig_s_plus_1p3b_list, marker='^', label='S/√(S+1.3B)')
            # ax_top.plot(cut_values, sig_binomial_list, marker='x', label='BinomialExpZ')
            ax_top.set_ylabel('Significance')
            ax_top.set_title(f'N-1: {cut_var_tmp} — Significance vs. {cut_var} ({cut_type})')
            ax_top.grid(True)
            ax_top.legend()

            # Bottom: (S/√B) × Acceptance
            if len(cut_values):
                ax_bot.axvline(x=max_tmp, color='r', linestyle='--')
            ax_bot.plot(cut_values, sigacc_simple_list, marker='o', label='(S/√B) × Acceptance')

            for i, acc in enumerate(acceptance_values):
                ax_bot.text(cut_values[i], sigacc_simple_list[i], f'{acc:.1f}%',
                            fontsize=9, ha='right', va='bottom', color='purple')

            # Label with pretty var title
            var_cfg_for_label = getVarDict(tot2[0], signal_name, cut_var)
            ax_bot.set_xlabel(var_cfg_for_label[cut_var]['title'])
            ax_bot.set_ylabel('Significance × Acceptance')
            ax_bot.grid(True)
            ax_bot.legend()

            plt.tight_layout()
            out_path = f"{out_base}/significance_{cut_var}_{cut_type}.png"
            ensure_dir(Path(out_path).parent.as_posix())
            plt.savefig(out_path)
            print(f"Saved: {out_path}")
            plt.close()

    # --- N-1 distributions + per-bin significance & ROC for THIS variable ---
    var_cfg_sig = getVarDict(tot2[0], signal_name, var_name=cut_var_tmp)  # only request this var
    for var in var_cfg_sig:
        bg_vals, bg_wts, bg_cols, bg_labs = [], [], [], []
        sig_vals, sig_wts = [], []
        sig_col = None
        sig_lab = None

        # Build stacks
        for j in range(len(ntuple_names)):
            process = ntuple_names[j]
            fb2 = tot2[j]
            var_cfg = getVarDict(fb2, process, var_name=var)
            x = var_cfg[var]['var']
            bins = var_cfg[var]['bins']
            weights = fb2['weights']

            info = sample_dict[process]
            if process == signal_name:
                sig_vals.append(x)
                sig_wts.append(weights)
                sig_col = info['color']; sig_lab = info['legend']
            else:
                bg_vals.append(x); bg_wts.append(weights)
                bg_cols.append(info['color']); bg_labs.append(info['legend'])

        # Convert/concat
        sig_all = safe_concat(sig_vals)
        sig_w_all = safe_concat(sig_wts)
        bg_all = safe_concat(bg_vals)
        bg_w_all = safe_concat(bg_wts)

        # Figure / axes
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 13), gridspec_kw={'height_ratios':[9,4]})

        # Stacked BG + signal outline
        if len(bg_vals):
            ax_top.hist([to_np(v) for v in bg_vals], bins=bins,
                        weights=[to_np(w) for w in bg_wts], color=bg_cols,
                        label=bg_labs, stacked=True)
        if sig_all.size:
            ax_top.hist(sig_all, bins=bins, weights=sig_w_all, color=sig_col,
                        label=sig_lab, histtype='step', linewidth=2)

            # Signal error bars
            s_counts, s_edges = safe_hist(sig_all, bins=bins, w=sig_w_all)
            s2, _ = safe_hist(sig_all, bins=bins, w=sig_w_all**2 if sig_w_all.size else None)
            bin_centers = 0.5*(s_edges[:-1] + s_edges[1:])
            s_err = np.sqrt(s2)
            ax_top.errorbar(bin_centers, s_counts, yerr=s_err, fmt='.', linewidth=2,
                            color=sig_col, capsize=0)

        ax_top.set_yscale('log')
        ax_top.set_ylim(max(1e-4, 1e-6), 1e11)
        ax_top.set_xlim(bins[0], bins[-1])
        ax_top.minorticks_on()
        ax_top.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_top.set_ylabel("Events")
        ax_top.legend(ncol=2)

        # Per-bin significance curves
        S_counts, _ = safe_hist(sig_all, bins=bins, w=sig_w_all)
        B_counts, _ = safe_hist(bg_all, bins=bins, w=bg_w_all)

        sqrt_B = np.sqrt(np.clip(B_counts, 0, None))
        sqrt_SplusB = np.sqrt(np.clip(S_counts + B_counts, 0, None))
        sqrt_Splus1p3B = np.sqrt(np.clip(S_counts + 1.3*B_counts, 0, None))

        sig_simple = np.where(B_counts > 0, S_counts / sqrt_B, 0.0)
        sig_s_plus_b = np.where((S_counts + B_counts) > 0, S_counts / sqrt_SplusB, 0.0)
        sig_s_plus_1p3b = np.where((S_counts + 1.3*B_counts) > 0, S_counts / sqrt_Splus1p3B, 0.0)

        # Binomial ExpZ per bin
        zbi_per_bin = np.array([zbi(S_counts[i], B_counts[i], sigma_b_frac=0.3) for i in range(len(S_counts))])

        bin_centers = 0.5*(bins[:-1] + bins[1:])

        # Totals
        S_tot = float(np.sum(S_counts))
        B_tot = float(np.sum(B_counts))
        if B_tot > 0:
            tot_SsqrtB = S_tot / np.sqrt(B_tot)
            tot_SsqrtSB = S_tot / np.sqrt(S_tot + B_tot) if (S_tot + B_tot) > 0 else 0
            tot_SsqrtS1p3B = S_tot / np.sqrt(S_tot + 1.3*B_tot) if (S_tot + 1.3*B_tot) > 0 else 0
            tot_zbi = zbi(S_tot, B_tot, sigma_b_frac=0.3)
        else:
            tot_SsqrtB = tot_SsqrtSB = tot_SsqrtS1p3B = tot_zbi = 0.0

        ax_bot.step(bin_centers, sig_simple, where='mid', linewidth=2,
                    label=f"S/√B = {tot_SsqrtB:.4f}", color='chocolate')
        ax_bot.step(bin_centers, sig_s_plus_b, where='mid', linewidth=2,
                    label=f"S/√(S+B) = {tot_SsqrtSB:.4f}", color='tomato')
        ax_bot.step(bin_centers, sig_s_plus_1p3b, where='mid', linewidth=2,
                    label=f"S/√(S+1.3B) = {tot_SsqrtS1p3B:.4f}", color='orange')
        ax_bot.step(bin_centers, zbi_per_bin, where='mid', linewidth=2,
                    label=f"Binomial ExpZ = {tot_zbi:.4f}", color='plum')

        ax_bot.set_xlabel(var_cfg_sig[var]['title'])
        ax_bot.set_ylabel("Significance")
        ax_bot.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_bot.set_title("")
        leg = ax_bot.legend()
        for t in leg.get_texts():
            t.set_color('purple')

        plt.xlim(bins[0], bins[-1])
        plt.tight_layout()

        dist_dir = f"{out_base}"
        ensure_dir(dist_dir)
        out_png = f"{dist_dir}/{var}.png"
        plt.savefig(out_png)
        print(f"Saved: {out_png}")
        plt.close()

        # ROC using this single variable as score
        y_true = np.concatenate([np.ones_like(S_counts).repeat(1)])  # placeholder to silence lints
        # Build event-wise arrays, not binned:
        y_true = np.concatenate([np.ones(sig_all.shape[0], dtype=int), np.zeros(bg_all.shape[0], dtype=int)])
        y_scores = np.concatenate([sig_all, bg_all])
        y_w = np.concatenate([sig_w_all if sig_w_all.size else np.ones(sig_all.shape[0]),
                              bg_w_all if bg_w_all.size else np.ones(bg_all.shape[0])])

        if y_scores.size and np.unique(y_true).size == 2:
            fpr, tpr, thr = roc_curve(y_true, y_scores, sample_weight=y_w)
            order = np.argsort(fpr)
            roc_auc = auc(fpr[order], tpr[order])

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.5f})')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"N-1 ROC — {var}")
            plt.legend(loc="lower right")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            out_png = f"{dist_dir}/roc_curve_{var}.png"
            plt.savefig(out_png)
            print(f"Saved: {out_png}")
            plt.close()
