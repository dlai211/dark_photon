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
from plot_config import getWeight, zbi, sample_dict, getVarDict
from plot_var import variables, ntuple_names, ntuple_names_BDT


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

def test(fb):
    # checking if there are any none values
    mask = ak.is_none(fb['met_tst_et'])
    n_none = ak.sum(mask)
    print("Number of none values: ", n_none)
    # if n_none > 0:
    #     fb = fb[~mask]
    # print("Events after removing none values: ", len(fb), ak.sum(ak.is_none(fb['met_tst_et'])))

# i = 0
for i in range(len(ntuple_names)-1):
    ucut, wcut = [], []
    start_time = time.time()
    ntuple_name = ntuple_names[i]
    path = f"/data/projects/campfire-workshop/dark_photon/ntups/mc23d/{ntuple_names[i]}_y.root" 
    path_BDT = f"/data/projects/campfire-workshop/dark_photon/ntups/withVertexBDT/mc23d_{ntuple_names_BDT[i]}_y_BDT_score.root" 
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
    

    print("Unweighted Events before cut: ", len(fb))
    print("Weighted Events before cut: ", sum(getWeight(fb, ntuple_name)))
    unweighted_bcut.append(len(fb))
    weighted_bcut.append(sum(getWeight(fb, ntuple_name)))


    fb = fb[fb['n_ph_baseline'] == 1]
    fb = fb[fb['n_ph'] == 1]
    fb = fb[fb['n_mu_baseline'] == 0]
    fb = fb[fb['n_el_baseline'] == 0]
    fb = fb[fb['n_tau_baseline'] == 0]
    fb = fb[fb['trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M']==1]
    fb = fb[ak.num(fb['ph_pt']) > 0] # prevent none values in Tbranch
    fb = fb[fb['met_tst_et'] >= 100000] # MET cut (basic cut)
    fb = fb[ak.firsts(fb['ph_pt']) >= 50000] # ph_pt cut (basic cut)
    fb = fb[fb['n_jet_central'] <= 4] # n_jet_central cut (basic cut)
    # goodPV on signal only
    if ntuple_name == 'ggHyyd':
        fb = fb[ak.num(fb['pv_z']) > 0]
        good_pv_tmp = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)
        fb = fb[good_pv_tmp]

    mt_tmp = np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) * 
                            (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000
    mask1 = mt_tmp >= 100 # trigger cut
    fb = fb[mask1]
    ucut.append(len(fb))
    wcut.append(sum(getWeight(fb, ntuple_name)))


    print("Unweighted Events after cut: ", len(fb))
    print("Weighted Events after cut: ", sum(getWeight(fb, ntuple_name)))
    ucut.append(len(fb))
    wcut.append(sum(getWeight(fb, ntuple_name)))

    unweighted_acut.append(ucut)
    # unweighted_acut.append(len(fb))
    weighted_acut.append(wcut)
    test(fb) # check for none value

    print(f"Reading Time for {ntuple_name}: {(time.time()-start_time)} seconds\n")



    tot.append(fb)

    fb = 0
    fb_BDT = 0
    tmp = 0


Vars = [
    'metsig',
    'metsigres',
    'met',
    'met_noJVT',
    'dmet',
    'ph_pt',
    'ph_eta',
    'ph_phi',
    'jet_central_eta',
    'jet_central_pt1',
    'jet_central_pt2',
    'dphi_met_phterm',
    'dphi_met_ph',
    'dphi_met_jetterm',
    'dphi_phterm_jetterm',
    'dphi_ph_centraljet1',
    'dphi_ph_jet1',
    'metplusph',
    'failJVT_jet_pt1',
    'softerm',
    'jetterm',
    'jetterm_sumet',
    'dphi_met_central_jet',
    'balance',
    'dphi_jj',
    'BDTScore',
    'n_jet_central'
]

data_list = []

for j in range(len(ntuple_names)-1):
    process = ntuple_names[j]
    fb = tot[j] 
    
    data_dict = {}
    
    for var in Vars:
        var_config = getVarDict(fb, process, var_name=var)
        data_dict[var] = var_config[var]['var']
    
    weights = getWeight(fb, process)
    data_dict['weights'] = weights
    data_dict['process'] = ntuple_names[j]
    
    n_events = len(weights)
    data_dict['process'] = [process] * n_events
    label = 1 if process == 'ggHyyd' else 0
    data_dict['label'] = [label] * n_events
    
    df_temp = pd.DataFrame(data_dict)
    data_list.append(df_temp)

df_all = pd.concat(data_list, ignore_index=True)
df_all.head()

df_all.to_csv("/data/projects/campfire-workshop/dark_photon/ML_input/BDT_input_basic1.csv", index=False)
