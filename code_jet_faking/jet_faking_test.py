# import modules
import uproot, sys, time, math
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


tot = []
data = pd.DataFrame()
unweighted_bcut, weighted_bcut, unweighted_acut, weighted_acut = [], [], [], []
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
    ucut, wcut = [], []
    start_time = time.time()
    ntuple_name = ntuple_names[i]
    if ntuple_name == 'data23': # data
        path = f"/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/data23_y_BDT_score.root" 
        print('processing file: ', path)
        f = uproot.open(path)['nominal']
        fb = f.arrays(variables_data, library="ak")
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
            fb["BDTScore"] = fb_BDT["BDTScore"]
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
    fb = fb[fb['n_jet_central'] <= 4] # n_jet_central cut (basic cut)

    mt_tmp = np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) * 
                            (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000
    mask1 = mt_tmp >= 100 # trigger cut
    fb = fb[mask1]

    # fb = fb[fb['BDTScore'] >= 0.1] # added cut 1
    

    print_cut(ntuple_name, fb, 'after basic cut')


    ucut.append(len(fb))

    unweighted_acut.append(ucut)
    weighted_acut.append(wcut)
    test(fb) # check for none value

    print(f"Reading Time for {ntuple_name}: {(time.time()-start_time)} seconds\n")



    tot.append(fb)

    fb = 0
    fb_BDT = 0
    tmp = 0
