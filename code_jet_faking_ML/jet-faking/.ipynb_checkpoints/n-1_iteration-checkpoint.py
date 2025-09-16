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

# import config functions
from jet_faking_plot_config import getWeight, zbi, sample_dict, getVarDict
from plot_var import variables, variables_data, ntuple_names, ntuple_names_BDT

# import cut config
from cut_config import cut_config

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
    "legend": 14
}

plt.rcParams.update({
    "axes.labelsize": font_size["xlabel"],  # X and Y axis labels
    "xtick.labelsize": font_size["xticks"],  # X ticks
    "ytick.labelsize": font_size["yticks"],  # Y ticks
    "legend.fontsize": font_size["legend"]  # Legend
})


print(' < -- Reading nTuples files & Applying basic cuts -- > ')
tot = []
data = pd.DataFrame()
ntuple_names = ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct', 'data23']

def test(fb):
    # checking if there are any none values
    mask = ak.is_none(fb['met_tst_et'])
    n_none = ak.sum(mask)
    print("Number of none values: ", n_none)
    # if n_none > 0:
    #     fb = fb[~mask]
    # print("Events after removing none values: ", len(fb), ak.sum(ak.is_none(fb['met_tst_et'])))

def print_cut(ntuple_name, fb, label):
    print(f"Unweighted Events {label}: ", len(fb))
    if ntuple_name == 'data23':
        print(f"Weighted Events {label}: ", sum(getWeight(fb, ntuple_name, jet_faking=True)))
    else: 
        print(f"Weighted Events {label}: ", sum(getWeight(fb, ntuple_name)))

for i in range(len(ntuple_names)):
    start_time = time.time()
    ntuple_name = ntuple_names[i]
    if ntuple_name == 'data23': # data
        path = f"/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/data23_y_BDT_score.root" 
        print('processing file: ', path)
        f = uproot.open(path)['nominal']
        fb = f.arrays(variables_data, library="ak")
        fb['VertexBDTScore'] = fb['BDTScore'] # renaming BDTScore to ensure this is recognized as Vertex BDT Score
        
        fb = fb[ak.num(fb['ph_eta']) > 0]     # for abs(ak.firsts(fb['ph_eta'])) to have value to the reweighting
                
        mask1 = (ak.firsts(fb['ph_topoetcone40'])-2450.)/ak.firsts(fb['ph_pt']) > 0.1   # jet_faking_photon cut
        fb = fb[mask1]
        fb = fb[fb['n_ph_baseline'] == 1]

    else: # MC
        path = f"/data/tmathew/ntups/mc23d/{ntuple_name}_y.root" 
        path_BDT = f"/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/mc23d_{ntuple_name}_y_BDT_score.root" 
        print('processing file: ', path)
        f = uproot.open(path)['nominal']
        fb = f.arrays(variables, library="ak")

        # add BDT score to fb
        f_BDT = uproot.open(path_BDT)['nominal']
        fb_BDT = f_BDT.arrays(["event", "BDTScore"], library="ak")
        tmp = fb["event"] == fb_BDT["event"]
        if np.all(tmp) == True:
            fb["VertexBDTScore"] = fb_BDT["BDTScore"]
        else: 
            print("Something is wrong, need arranging")

        fb = fb[ak.num(fb['ph_eta']) > 0]     # for abs(ak.firsts(fb['ph_eta'])) to have value to the reweighting
        fb = fb[fb['n_ph'] == 1]
        
        # Zjets and Wjets (rule out everything except for e->gamma)
        if ntuple_name == 'Zjets' or ntuple_name == 'Wjets':
            mask = ak.firsts(fb['ph_truth_type']) == 2
            fb = fb[mask]
        
        # goodPV on signal only
        if ntuple_name == 'ggHyyd':
            fb = fb[ak.num(fb['pv_z']) > 0]
            good_pv_tmp = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)
            fb = fb[good_pv_tmp]

    print_cut(ntuple_name, fb, 'before cut')

    fb = fb[fb['n_mu_baseline'] == 0]
    fb = fb[fb['n_el_baseline'] == 0]
    fb = fb[fb['n_tau_baseline'] == 0]
    fb = fb[fb['trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M']==1]
    fb = fb[ak.num(fb['ph_pt']) > 0] # prevent none values in Tbranch
    fb = fb[ak.firsts(fb['ph_pt']) >= 50000] # ph_pt cut (basic cut)
    fb = fb[fb['met_tst_et'] >= 100000] # MET cut (basic cut)
    fb = fb[fb['n_jet_central'] <= 3] # n_jet_central cut (basic cut)

    mt_tmp = np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) * 
                            (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000
    mask1 = mt_tmp > 100 # trigger cut
    mask2 = mt_tmp < 140
    fb = fb[mask1 * mask2]

    fb = fb[fb['VertexBDTScore'] > 0.1]

    print_cut(ntuple_name, fb, 'after basic cut')

    print(f"Reading Time for {ntuple_name}: {(time.time()-start_time)} seconds\n")

    tot.append(fb)

    fb = 0
    fb_BDT = 0
    tmp = 0



signal_name = 'ggHyyd'  # Define signal dataset
cut_name = 'basic'


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
            mask = x != -999
            x = x[mask]

            if process == signal_name:
                sig_events = getWeight(fb, process)
                sig_events = sig_events[mask]
                mask = x >= cut if cut_type == 'lowercut' else x <= cut
                sig_after_cut = ak.sum(sig_events[mask])
            else:
                bkg_events = getWeight(fb, process)
                bkg_events = bkg_events[mask]
                mask = x >= cut if cut_type == 'lowercut' else x <= cut
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

# os.makedirs("plot_data", exist_ok=True)
initial_cut = []
tot2 = tot

print( ' < -- Initial Cut on all variables (maximize the significance * acceptance) -- > ')
start_time1 = time.time()
for cut_var, cut_types in cut_config.items():
    for cut_type, cut_values in cut_types.items():
        sig_simple_list, sigacc_simple_list, acceptance_values = calculate_significance(
            cut_var, cut_type, cut_values, tot2, ntuple_names, signal_name, getVarDict, getWeight
        )

        best_cut, best_sig, idx = get_best_cut(cut_values, sigacc_simple_list) 
        
        if idx == 0 or idx == len(sigacc_simple_list) - 1: # I chose to use index to indicate not to make unnecessary cut (for initial cut)
            print(cut_var, idx, len(sigacc_simple_list))
            continue
            
        result = {
            "cut_var": cut_var,
            "cut_type": cut_type,
            "best_cut": best_cut,
            "best_sig_x_acc": best_sig,
            "significance": sig_simple_list[idx],
            "acceptance": acceptance_values[idx]
        }

        print(result)
        initial_cut.append(dict(list(result.items())[:3]))


def apply_cut_to_fb(fb, process, var, cut_val, cut_type, getVarDict):
    var_config = getVarDict(fb, process, var_name=var)
    x = var_config[var]['var']
    mask = x != -999

    if cut_type == 'lowercut':
        mask = mask & (x >= cut_val)
    elif cut_type == 'uppercut':
        mask = mask & (x <= cut_val)

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

tot2_initial_cut = apply_all_cuts(tot2, ntuple_names, initial_cut, getVarDict)
final_significance = compute_total_significance(tot2_initial_cut, ntuple_names, signal_name, getVarDict, getWeight)
print('After initial cutting, signficance: ', final_significance)
print(f"Initial cutting time: {(time.time() - start_time1)/60} minutes\n")



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

print( ' < -- n-1 iterations until no further improvement (max significance) -- > ')
start_time2 = time.time()
optimized_cuts, final_significance = n_minus_1_optimizer(
    initial_cut, cut_config, tot2, ntuple_names, signal_name, getVarDict, getWeight, final_significance
)
print('After optimized cutting, signficance: ', final_significance)
print(f"Optimized cutting time: {(time.time() - start_time2)/60} minutes\n")


print( ' < -- Final Optimized Cuts -- > ')
print(optimized_cuts)


















