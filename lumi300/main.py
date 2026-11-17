# import modules
import uproot, sys, time, math, pickle, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from scipy.special import betainc
from scipy.stats import norm
from pathlib import Path

# import config functions
sys.path.append('/home/jlai/dark_photon/code/config')
from plot_config import getWeight, zbi, sample_dict, getVarDict
from plot_var import variables, variables_mc, ntuple_names
from n_1_iteration_functions import get_best_cut, calculate_significance, apply_cut_to_fb, apply_all_cuts, compute_total_significance, n_minus_1_optimizer
from perf_sig_plot import plot_performance, plot_significance, plot_n_1, calculate_significance2

# Set up plot defaults
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14.0,10.0  # Roughly 11 cm wde by 8 cm high  
mpl.rcParams['font.size'] = 20.0 # Use 14 point font
sns.set(style="whitegrid")

font_size = {
    "xlabel": 17,
    "ylabel": 17,
    "xticks": 15,
    "yticks": 15,
    "legend": 14,
    "title": 20
}

plt.rcParams.update({
    "axes.labelsize": font_size["xlabel"],  # X and Y axis labels
    "xtick.labelsize": font_size["xticks"],  # X ticks
    "ytick.labelsize": font_size["yticks"],  # Y ticks
    "legend.fontsize": font_size["legend"],  # Legend
    "axes.titlesize": font_size["title"] # Title
})

tot = []
signal_name = 'ggHyyd'
data = pd.DataFrame()
# additional_variables = ['MLBDTScore', 'MLBDTScore2', 'MLNNScore']
additional_variables = ['']
# ntuple_names = ntuple_names + ["mc23d_VBFHyyd_y", "mc23d_WHyyd_y", "mc23d_ZHyyd_y", "mc23e_VBFHyyd_y", "mc23e_WHyyd_y", "mc23e_ZHyyd_y"]
ntuple_names = ['mc23d_ggHyyd_y', 'mc23d_gammajet_direct_y', 'mc23e_ggHyyd_y', 'mc23e_gammajet_direct_y'] 

def test(fb):
    # checking if there are any none values
    mask = ak.is_none(fb['met_tst_et'])
    n_none = ak.sum(mask)
    print("Number of none values: ", n_none)
    # if n_none > 0:
    #     fb = fb[~mask]
    # print("Events after removing none values: ", len(fb), ak.sum(ak.is_none(fb['met_tst_et'])))

def print_cut(ntuple_name, fb, label):
    print(f"{ntuple_name} Unweighted Events {label}: ", len(fb))
    print(f"{ntuple_name} Weighted Events {label}: ", sum(getWeight(fb, ntuple_name)))
        
for i in range(len(ntuple_names)):
    start_time = time.time()
    ntuple_name = ntuple_names[i]
    # path = f"/data/fpiazza/ggHyyd/NtuplesWithBDTSkim/{ntuple_name}_nominal_bdt.root"
    path = f"/data/jlai/df_ntups_w_bdt_new/{ntuple_name}_nominal_bdt_withMLScores.root"
    f = uproot.open(path)['nominal']
    if ntuple_name.startswith("mc"):
        fb = f.arrays(variables_mc, library='ak')
        print_cut(ntuple_name, fb, 'before cut')
        
        fb = fb[ak.num(fb['ph_eta']) > 0]     # for abs(ak.firsts(fb['ph_eta'])) to have value to the reweighting
        fb = fb[fb['n_ph'] == 1]
        fb = fb[fb['n_el_baseline'] == 0]
        mask1 = (ak.firsts(fb['ph_topoetcone40'])-2450.)/ak.firsts(fb['ph_pt']) < 0.022
        fb = fb[mask1]

        # if 'ggHyyd' in ntuple_name: # goodPV on signal only
        #     fb = fb[ak.num(fb['pv_z']) > 0]
        #     good_pv_tmp = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)
        #     fb = fb[good_pv_tmp]
            
        
    if (ntuple_name == "data23_y") or (ntuple_name == "data24_y"):  # jet-faking 
        fb = f.arrays(variables, library='ak')
        print_cut(ntuple_name, fb, 'before cut')

        fb = fb[ak.num(fb['ph_eta']) > 0]
        mask1 = (ak.firsts(fb['ph_topoetcone40'])-2450.)/ak.firsts(fb['ph_pt']) > 0.1   # jet_faking_photon cut
        fb = fb[mask1]
        fb = fb[fb['n_ph_baseline'] == 1]
        fb = fb[fb['n_el_baseline'] == 0]


    if (ntuple_name == "data23_eprobe") or (ntuple_name == "data24_eprobe"): # electron-faking
        fb = f.arrays(variables+['trigger_single_el'], library='ak')
        print_cut(ntuple_name, fb, 'before cut')
        
        fb = fb[fb['n_el'] == 1]
        fb = fb[fb['n_ph_baseline'] == 0]

        # using electron info for photon info
        fb['ph_pt'] = fb['el_pt']
        fb['ph_eta'] = fb['el_eta']
        fb['ph_phi'] = fb['el_phi']
        fb['dphi_met_phterm'] = fb['dphi_met_eleterm']  

        # trigger
        fb = fb[fb['trigger_single_el'] == 1]


    fb = fb[ak.num(fb['ph_pt']) > 0] # prevent none values in Tbranch
    fb = fb[ak.firsts(fb['ph_pt']) >= 50000] # ph_pt cut (Preselection cut)
    fb = fb[fb['n_mu_baseline'] == 0]
    fb = fb[fb['n_tau_baseline'] == 0]
    fb = fb[fb['n_bjet'] == 0]
    fb = fb[fb['trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M']==1]
    fb = fb[fb['met_tst_et'] >= 100000] # MET cut (Preselection cut)
    fb = fb[fb['n_jet_central'] <= 3] # n_jet_central cut (Preselection cut)
    
    fb['VertexBDTScore'] = fb['BDTScore'] # renaming BDTScore to ensure this is recognized as Vertex BDT Score
    # fb = fb[fb['VertexBDTScore'] > 0.1]
    
    mt_tmp = np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) * 
                    (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000
    # mask1 = mt_tmp > 100
    # mask2 = mt_tmp < 140
    # fb = fb[mask1 * mask2]
    mask = mt_tmp > 80
    fb = fb[mask]



    # ------ Adjustment --------
    fb['weights'] = getWeight(fb, ntuple_name)


    dphi_met_jetterm_tmp = fb['dphi_met_jetterm']
    dphi_met_jetterm_tmp = ak.where(dphi_met_jetterm_tmp == -10, np.nan, dphi_met_jetterm_tmp)
    dphi_met_jetterm_tmp = np.arccos(np.cos(dphi_met_jetterm_tmp))
    fb['dphi_met_jetterm'] = ak.where(np.isnan(dphi_met_jetterm_tmp), -999, dphi_met_jetterm_tmp)


    dphi_met_phterm_tmp = fb['dphi_met_phterm']
    dphi_met_phterm_tmp = ak.where(dphi_met_phterm_tmp == -10, np.nan, dphi_met_phterm_tmp)
    dphi_met_phterm_tmp = np.arccos(np.cos(dphi_met_phterm_tmp))
    fb['dphi_met_phterm'] = ak.where(np.isnan(dphi_met_phterm_tmp), -999, dphi_met_phterm_tmp)
    print(min(fb['dphi_met_phterm']))

    fb['is_good_pv'] = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)

    print_cut(ntuple_name, fb, 'after preselection cuts')

    # fb = fb[ak.num(fb['pv_z']) > 0]
    # good_pv_tmp = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)
    # fb = fb[good_pv_tmp]
    # print_cut(ntuple_name, fb, "after good pv")
    
    test(fb) # check for none value

    print(f"Reading Time for {ntuple_name}: {(time.time()-start_time)} seconds\n")

    tot.append(fb)

    del fb 


# combining 23d + 23e {ggHyyd (0, 2), gammajet_direct (1, 3)}

tot_tmp = tot
tot2 = []
for i in tqdm(range(2)):
    tot2.append(ak.concatenate([tot_tmp[i], tot_tmp[i+2]]))
ntuple_names = ["ggHyyd", "gammajet_direct"]
del tot_tmp


def getVarDict(fb, process, var_name=None):
    var_dict = {}
    if var_name is None or var_name == 'dmet':
        var_dict['dmet'] = {
            'var': fb['dmet'],
            'bins': np.linspace(-100000, 100000, 20+1),
            'title': r'$E_{T,\mathrm{noJVT}}^{miss}-E_T^{miss}\ [MeV]$'
        }
    if var_name is None or var_name == 'met':
        var_dict['met'] = {
            'var': fb['met_tst_et'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ [MeV]$'
        }
    if var_name is None or var_name == 'dphi_met_phterm':
        var_dict['dphi_met_phterm'] = {
            'var': fb['dphi_met_phterm'],
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})$',
        }
    if var_name is None or var_name == 'metsig':
        var_dict['metsig'] = {
            'var': fb['met_tst_sig'],
            'bins': np.linspace(0, 30, 15+1),
            'title': r'$E_T^{miss}\ significance$'
        }
    return var_dict


def scale_good_bad_weights(w_good, w_bad, pct, mode="migration"):
    """
    - mode="migration": Preserves total yield. Scales good vertices up and bad vertices down.
    - mode="rejection": Leaves Good PVs untouched. Shrinks bad vertices simulating them failing the MET cut.
    
    pct: Percentage of Bad PVs that are successfully fixed (0 to 100).
         0% = Baseline (no changes).
         100% = All Bad PVs are fixed.
    """
    P = pct / 100.0
    sum_g = np.sum(w_good)
    sum_b = np.sum(w_bad)
    
    if sum_g + sum_b == 0:
        return w_good, w_bad

    if mode == "migration":
        # We migrate P fraction of Bad PVs into the Good PV bin
        target_g = sum_g + (sum_b * P)
        target_b = sum_b * (1.0 - P)
        
        scale_g = (target_g / sum_g) if sum_g > 0 else 0.0
        scale_b = (target_b / sum_b) if sum_b > 0 else 0.0
        
    elif mode == "rejection":
        # The P fraction of Bad PVs loses its fake MET and disappears
        scale_g = 1.0          
        scale_b = 1.0 - P      

    return w_good * scale_g, w_bad * scale_b


def generate_variable_plots(tot_array, apply_vertex_bdt=False, variables_to_plot=['dmet', 'met', 'dphi_met_phterm', 'metsig']):
    folder_tag = "with_vertexbdt" if apply_vertex_bdt else "without_vertexbdt"
    base_out_dir = f"/home/jlai/dark_photon/lumi300/plots_{folder_tag}"
    os.makedirs(base_out_dir, exist_ok=True)

    fb_sig = tot_array[0]
    fb_bkg = tot_array[1]

    if apply_vertex_bdt:
        fb_sig = fb_sig[fb_sig['VertexBDTScore'] > 0.1]
        fb_bkg = fb_bkg[fb_bkg['VertexBDTScore'] > 0.1]

    modes = [
        ("both_mig", "migration", "migration"),
        ("sig_mig_bkg_rej", "migration", "rejection")
    ]
    pcts = range(0, 101, 10)

    for var in variables_to_plot:
        var_config = getVarDict(fb_sig, 'ggHyyd', var_name=var)
        bins = var_config[var]['bins']
        title = var_config[var]['title']

        sig_x = ak.to_numpy(var_config[var]['var'])
        sig_w = ak.to_numpy(fb_sig['weights'])
        sig_good = ak.to_numpy(fb_sig['is_good_pv'])

        bkg_config = getVarDict(fb_bkg, 'gammajet_direct', var_name=var)
        bkg_x = ak.to_numpy(bkg_config[var]['var'])
        bkg_w = ak.to_numpy(fb_bkg['weights'])
        bkg_good = ak.to_numpy(fb_bkg['is_good_pv'])

        sig_x_good, sig_w_good_0 = sig_x[sig_good], sig_w[sig_good]
        sig_x_bad,  sig_w_bad_0  = sig_x[~sig_good], sig_w[~sig_good]
        bkg_x_good, bkg_w_good_0 = bkg_x[bkg_good], bkg_w[bkg_good]
        bkg_x_bad,  bkg_w_bad_0  = bkg_x[~bkg_good], bkg_w[~bkg_good]

        colors = {"bkg_good": "#4C72B0", "bkg_bad": "#DD8452", "sig_good": "#D62728", "sig_bad": "#9467BD"}

        print(f"[{folder_tag}] Generating plots for {var}...")
        for (mode_name, sig_mode, bkg_mode) in modes:
            for sig_pct in pcts:
                for bkg_pct in pcts:
                    
                    sig_w_good, sig_w_bad = scale_good_bad_weights(sig_w_good_0, sig_w_bad_0, sig_pct, mode=sig_mode)
                    bkg_w_good, bkg_w_bad = scale_good_bad_weights(bkg_w_good_0, bkg_w_bad_0, bkg_pct, mode=bkg_mode)

                    sig_all_x = np.concatenate([sig_x_good, sig_x_bad])
                    sig_all_w = np.concatenate([sig_w_good, sig_w_bad])
                    bkg_all_x = np.concatenate([bkg_x_good, bkg_x_bad])
                    bkg_all_w = np.concatenate([bkg_w_good, bkg_w_bad])

                    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 13), gridspec_kw={'height_ratios': [9, 4]})

                    ax_top.hist([bkg_x_bad, bkg_x_good], bins=bins, weights=[bkg_w_bad, bkg_w_good],
                                color=[colors["bkg_bad"], colors["bkg_good"]],
                                label=[rf'$\gamma$+jets (Bad PV) [$\Sigma w$: {np.sum(bkg_w_bad):.0f}]', 
                                       rf'$\gamma$+jets (Good PV) [$\Sigma w$: {np.sum(bkg_w_good):.0f}]'],
                                stacked=True, alpha=0.85)

                    ax_top.hist(sig_x_good, bins=bins, weights=sig_w_good, color=colors["sig_good"], 
                                label=rf'ggHyyd (Good PV) [$\Sigma w$: {np.sum(sig_w_good):.1f}]',
                                histtype='step', linewidth=2.5)
                    ax_top.hist(sig_x_bad, bins=bins, weights=sig_w_bad, color=colors["sig_bad"],
                                label=rf'ggHyyd (Bad PV) [$\Sigma w$: {np.sum(sig_w_bad):.1f}]',
                                histtype='step', linewidth=2.5, linestyle='--')

                    for sx, sw, scol in [(sig_x_good, sig_w_good, colors["sig_good"]), (sig_x_bad, sig_w_bad, colors["sig_bad"])]:
                        if len(sx) > 0 and np.sum(sw) > 0:
                            s_counts, bin_edges = np.histogram(sx, bins=bins, weights=sw)
                            sum_w_sq, _ = np.histogram(sx, bins=bins, weights=sw**2)
                            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                            ax_top.errorbar(bin_centers, s_counts, yerr=np.sqrt(sum_w_sq), fmt='.', linewidth=1.5, color=scol, capsize=0)

                    ax_top.set_yscale('log')
                    ax_top.set_ylim(1e-4, 1e11)
                    ax_top.set_xlim(bins[0], bins[-1])
                    ax_top.minorticks_on()
                    ax_top.grid(True, which="both", linestyle="--", linewidth=0.5)
                    ax_top.set_ylabel("Events")
                    ax_top.set_title(f"{var} | Sig PV: {sig_pct}% | Bkg PV: {bkg_pct}% | {mode_name}")
                    ax_top.legend(ncol=2)

                    S_counts, _ = np.histogram(sig_all_x, bins=bins, weights=sig_all_w)
                    B_counts, _ = np.histogram(bkg_all_x, bins=bins, weights=bkg_all_w)

                    sqrt_B = np.sqrt(B_counts)
                    sqrt_SplusB = np.sqrt(S_counts + B_counts)
                    sqrt_Splus1p3B = np.sqrt(S_counts + 1.3 * B_counts)

                    sig_simple = np.where(B_counts > 0, S_counts / sqrt_B, 0)
                    sig_s_plus_b = np.where((S_counts + B_counts) > 0, S_counts / sqrt_SplusB, 0)
                    sig_s_plus_1p3b = np.where((S_counts + 1.3 * B_counts) > 0, S_counts / sqrt_Splus1p3B, 0)
                    
                    try:
                        zbi_per_bin = np.array([zbi(S_counts[i], B_counts[i], sigma_b_frac=0.3) for i in range(len(S_counts))])
                    except NameError:
                        zbi_per_bin = sig_s_plus_1p3b

                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    
                    total_signal, total_bkg = np.sum(sig_all_w), np.sum(bkg_all_w)

                    if total_bkg > 0 and total_signal > 0:
                        total_sig_simple = total_signal / np.sqrt(total_bkg)
                        total_sig_s_plus_b = total_signal / np.sqrt(total_signal + total_bkg)
                        total_sig_s_plus_1p3b = total_signal / np.sqrt(total_signal + 1.3 * total_bkg)
                        try:
                            total_sig_binomial = zbi(total_signal, total_bkg, sigma_b_frac=0.3)
                        except NameError:
                            total_sig_binomial = total_sig_s_plus_1p3b
                    else:
                        total_sig_simple = total_sig_s_plus_b = total_sig_s_plus_1p3b = total_sig_binomial = 0.0

                    ax_bot.step(bin_centers, sig_simple, where='mid', color='chocolate', linewidth=2, label=f"S/√B = {total_sig_simple:.4f}")
                    ax_bot.step(bin_centers, sig_s_plus_b, where='mid', color='tomato', linewidth=2, label=f"S/√(S+B) = {total_sig_s_plus_b:.4f}")
                    ax_bot.step(bin_centers, sig_s_plus_1p3b, where='mid', color='orange', linewidth=2, label=f"S/√(S+1.3B) = {total_sig_s_plus_1p3b:.4f}")
                    ax_bot.step(bin_centers, zbi_per_bin, where='mid', color='plum', linewidth=2, label=f"Binomial ExpZ = {total_sig_binomial:.4f}")

                    ax_bot.set_xlabel(title)
                    ax_bot.set_ylabel("Significance")
                    ax_bot.set_xlim(bins[0], bins[-1])
                    ax_bot.grid(True, which="both", linestyle="--", linewidth=0.5)
                    leg = ax_bot.legend()
                    for text in leg.get_texts(): text.set_color('purple')
                    plt.tight_layout()

                    file_tag = f"{var}_{mode_name}_sig{sig_pct}_bkg{bkg_pct}"
                    plt.savefig(f"{base_out_dir}/{file_tag}.png", dpi=100)
                    plt.close()
                    print(f"successfully saved {base_out_dir}/{file_tag}.png")

                    if len(sig_all_x) > 0 and len(bkg_all_x) > 0 and np.sum(sig_all_w) > 0 and np.sum(bkg_all_w) > 0:
                        y_true = np.concatenate([np.ones_like(sig_all_x), np.zeros_like(bkg_all_x)])
                        if var == 'dmet':
                            y_scores = -np.abs(np.concatenate([sig_all_x, bkg_all_x]))
                        else:
                            y_scores = np.concatenate([sig_all_x, bkg_all_x])
                        y_weights = np.concatenate([sig_all_w, bkg_all_w])

                        fpr, tpr, _ = roc_curve(y_true, y_scores, sample_weight=y_weights)
                        sorted_idx = np.argsort(fpr)
                        roc_auc = auc(fpr[sorted_idx], tpr[sorted_idx])

                        plt.figure(figsize=(10, 10))
                        plt.plot(fpr[sorted_idx], tpr[sorted_idx], lw=2, color='red', label=f'ROC curve (AUC = {roc_auc:.5f})')
                        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random chance')
                        plt.xlabel("False Positive Rate (Background Efficiency)")
                        plt.ylabel("True Positive Rate (Signal Efficiency)")
                        plt.title(f"ROC | Sig: {sig_pct}% Bkg: {bkg_pct}% | {mode_name}")
                        plt.legend(loc="lower right")
                        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                        plt.tight_layout()
                        plt.savefig(f"{base_out_dir}/roc_{file_tag}.png", dpi=100)
                        plt.close()


print("Running phase space generation without BDT cut...")
generate_variable_plots(
    tot_array=tot2, 
    apply_vertex_bdt=False, 
    variables_to_plot=['dmet', 'met', 'dphi_met_phterm', 'metsig']
)

print("Running phase space generation with BDT cut...")
generate_variable_plots(
    tot_array=tot2, 
    apply_vertex_bdt=True, 
    variables_to_plot=['dmet', 'met', 'dphi_met_phterm', 'metsig']
)