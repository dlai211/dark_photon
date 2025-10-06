import numpy as np
import awkward as ak
def get_best_cut(cut_values, significance_list):
    max_idx = np.argmax(significance_list)
    best_cut = cut_values[max_idx]
    best_sig = significance_list[max_idx]
    return best_cut, best_sig, max_idx

def calculate_significance(cut_var, cut_type, cut_values, tot2, ntuple_names, signal_name, getVarDict, getWeight):
    sig_simple_list = []
    sigacc_simple_list = []
    acceptance_values = []
    tot_tmp = []

    for cut in cut_values:
        sig_after_cut = 0
        bkg_after_cut = []
        sig_events = 0

        for i in range(len(ntuple_names)):
            fb = tot2[i]
            process = ntuple_names[i]
            var_config = getVarDict(fb, process, var_name=cut_var)
            x = var_config[cut_var]['var']
            mask_nan = x == -999
            
            if process == signal_name:
                sig_events = getWeight(fb, process)
                mask_cut = x > cut if cut_type == 'lowercut' else x < cut
                mask = mask_nan | mask_cut
                sig_after_cut = ak.sum(sig_events[mask])
            else:
                bkg_events = getWeight(fb, process)
                mask_cut = x > cut if cut_type == 'lowercut' else x < cut
                mask = mask_nan | mask_cut
                bkg_after_cut.append(ak.sum(bkg_events[mask]))
            
            tot_tmp.append(fb)

        total_bkg = sum(bkg_after_cut)
        total_signal = sig_after_cut

        sig_simple = total_signal / np.sqrt(total_bkg) if total_bkg > 0 else 0
        acceptance = total_signal / sum(sig_events) if sum(sig_events) > 0 else 0

        sig_simple_list.append(sig_simple)
        sigacc_simple_list.append(sig_simple * acceptance)
        acceptance_values.append(acceptance * 100)

    return sig_simple_list, sigacc_simple_list, acceptance_values

def apply_cut_to_fb(fb, process, var, cut_val, cut_type, getVarDict):
    var_config = getVarDict(fb, process, var_name=var)
    x = var_config[var]['var']
    mask = x == -999

    if cut_type == 'lowercut':
        mask = mask | (x > cut_val)
    elif cut_type == 'uppercut':
        mask = mask | (x < cut_val)

    return fb[mask]

def apply_all_cuts(tot2, ntuple_names, cut_list, getVarDict):
    new_tot2 = []
    for i, fb in enumerate(tot2):
        process = ntuple_names[i]
        for cut in cut_list:
            fb = apply_cut_to_fb(fb, process, cut["cut_var"], cut["best_cut"], cut["cut_type"], getVarDict)
        new_tot2.append(fb)
    return new_tot2
    
def compute_total_significance(tot2, ntuple_names, signal_name, getVarDict, getWeight):
    signal_sum = 0
    bkg_sum = 0
    for i in range(len(ntuple_names)):
        fb = tot2[i]
        process = ntuple_names[i]
        weights = getWeight(fb, process)
        if process == signal_name:
            signal_sum += ak.sum(weights)
        else:
            bkg_sum += ak.sum(weights)
    return signal_sum / np.sqrt(bkg_sum) if bkg_sum > 0 else 0

def n_minus_1_optimizer(initial_cut, cut_config, tot2, ntuple_names, signal_name, getVarDict, getWeight, final_significance, max_iter=10, tolerance=1e-4):
    best_cuts = initial_cut.copy()
    iteration = 0
    converged = False

    while not converged and iteration < max_iter:
        converged = True
        print(f"\n--- Iteration {iteration + 1} ---")
        for i, cut in enumerate(best_cuts):
            # Apply all other cuts
            n_minus_1_cuts = best_cuts[:i] + best_cuts[i+1:]
            tot2_cut = apply_all_cuts(tot2, ntuple_names, n_minus_1_cuts, getVarDict)

            # Re-scan this variable
            cut_var = cut["cut_var"]
            cut_type = cut["cut_type"]
            cut_values = cut_config[cut_var][cut_type]

            sig_simple_list, sigacc_simple_list, _ = calculate_significance(
                cut_var, cut_type, cut_values, tot2_cut, ntuple_names
                , signal_name, getVarDict, getWeight
            )
            best_cut, best_sig, idx = get_best_cut(cut_values, sig_simple_list)

            if abs(best_cut - cut["best_cut"]) > tolerance:
            # if best_sig - final_significance > tolerance:
                print(f"Updating {cut_var} ({cut_type}): {cut['best_cut']} → {best_cut}  (sig {final_significance:.2f} → {best_sig:.2f})")
                best_cuts[i]["best_cut"] = best_cut
                final_significance = best_sig
                converged = False  # Found at least one improvement

        iteration += 1

    print( ' optimized cuts, end of iteration ' )
    return best_cuts, final_significance