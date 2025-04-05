
# script to save the signal and background in root file after selection cuts.

# import modules
import uproot, sys, time, ROOT
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

#import files
from config import variables, ntuple_name, ntuple_name_BDT
from utils import getWeight, getCutDict, getSampleDict, getVarDict, anyNone

if __name__ == '__main__':

    f_signal_out = ROOT.TFile("/data/jlai/ntups/mc23d/ggHyyd_y_selected.root", "RECREATE")
    f_bkg_out = ROOT.TFile("/data/jlai/ntups/mc23d/background_y_selected.root", "RECREATE")

    tree_signal_out = None 
    tree_bkg_out = None

    for i, name in enumerate(ntuple_name):
        path = f"/data/tmathew/ntups/mc23d/{name}_y.root" 
        f_in = ROOT.TFile.Open(path, "READ")
        t_in = f_in.Get("nominal")

        path_BDT = f"/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/mc23d_{ntuple_name_BDT[i]}_y_BDT_score.root" 
        f_in_BDT = ROOT.TFile.Open(path_BDT, "READ")
        t_in_BDT = f_in_BDT.Get("nominal")


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
        print("Weighted Events before cut: ", sum(getWeight(fb, ntuple_name[i])))
        unweighted_bcut.append(len(fb))
        weighted_bcut.append(sum(getWeight(fb, ntuple_name[i])))


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
        if ntuple_name[i] == 'ggHyyd':
            fb = fb[ak.num(fb['pv_z']) > 0]
            good_pv_tmp = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)
            fb = fb[good_pv_tmp]

        mt_tmp = np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) * 
                                (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000
        mask1 = mt_tmp >= 80 # trigger cut
        fb = fb[mask1]
        cut.append(len(fb))

        fb = fb[fb['BDTScore'] >= 0.1] # added cut 1
        cut.append(len(fb))

        metsig_tmp = fb['met_tst_sig'] # added cut 2 
        mask1 = metsig_tmp >= 7
        mask2 = metsig_tmp <= 16
        fb = fb[mask1 * mask2]
        cut.append(len(fb))

        dphi_met_phterm_tmp = np.arccos(np.cos(fb['met_tst_phi'] - fb['met_phterm_phi'])) # added cut 3
        fb = fb[dphi_met_phterm_tmp >= 1.3]
        cut.append(len(fb))

        dmet_tmp = fb['met_tst_noJVT_et'] - fb['met_tst_et'] # added cut 4
        mask1 = dmet_tmp >= -20000
        mask2 = dmet_tmp <= 50000
        fb = fb[mask1 * mask2]
        cut.append(len(fb))

        dphi_met_jetterm_tmp = np.where(fb['met_jetterm_et'] != 0,   # added cut 5
                                np.arccos(np.cos(fb['met_tst_phi'] - fb['met_jetterm_phi'])),
                                -999)
        fb = fb[dphi_met_jetterm_tmp <= 0.75]

        ph_eta_tmp = np.abs(ak.firsts(fb['ph_eta'])) # added cut 6
        fb = fb[ph_eta_tmp <= 1.75]
        
        # dphi_ph_centraljet1_tmp = np.arccos(np.cos(ak.firsts(fb['ph_phi']) - ak.firsts(fb['jet_central_phi']))) # added cut 4
        # dphi_ph_centraljet1_tmp = ak.fill_none(dphi_ph_centraljet1_tmp, -999)
        # valid_mask = dphi_ph_centraljet1_tmp != -999 # keeping -999 values
        # dphi_ph_centraljet1 = ak.mask(dphi_ph_centraljet1_tmp, (dphi_ph_centraljet1_tmp >= 1.5) | ~valid_mask)
        # fb = fb[~ak.is_none(dphi_ph_centraljet1)]
        # cut.append(len(fb))

        phi1_tmp = ak.firsts(fb['jet_central_phi']) # added cut 7
        phi2_tmp = ak.mask(fb['jet_central_phi'], ak.num(fb['jet_central_phi']) >= 2)[:, 1] 
        dphi_tmp = np.arccos(np.cos(phi1_tmp - phi2_tmp))
        dphi_jj_tmp = ak.fill_none(dphi_tmp, -1)
        fb = fb[dphi_jj_tmp <= 2.5]
        cut.append(len(fb))


        print("Unweighted Events after cut: ", len(fb))
        print("Weighted Events after cut: ", sum(getWeight(fb, ntuple_name[i])))
        cut.append(len(fb))

        unweighted_acut.append(cut)
        # unweighted_acut.append(len(fb))
        weighted_acut.append(sum(getWeight(fb, ntuple_name[i])))
        anyNone(fb) # check for none value

        print(f"Reading Time for {ntuple_name[i]}: {(time.time()-start_time)} seconds\n")



        tot.append(fb)

        fb = 0
        fb_BDT = 0
        tmp = 0





