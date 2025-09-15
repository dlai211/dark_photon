# import modules
import uproot, sys, time, math, pickle, os, csv, shutil
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
from datetime import datetime

# import config functions
# from jet_faking_plot_config import getWeight, zbi, sample_dict, getVarDict # 135fb^-1
from jet_faking_26_config import getWeight, zbi, sample_dict, getVarDict # 26fb^-1
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


# -------- CONFIG --------
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
# LOG_DIR = f"./cutlogs_{RUN_TAG}"
LOG_DIR = "./cutlogs_jet_faking_26_internal"
try:
    shutil.rmtree(LOG_DIR)
except FileNotFoundError:
    pass
os.makedirs(LOG_DIR, exist_ok=False)
TXT_LOG = os.path.join(LOG_DIR, "cutflow.log")
CSV_LOG = os.path.join(LOG_DIR, "cutflow.csv")

ntuple_names = ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct', 'data23']

def weight_sum(fb, ntuple_name):
    if ntuple_name == 'data23':
        return float(np.sum(getWeight(fb, ntuple_name, jet_faking=True)))
    else:
        return float(np.sum(getWeight(fb, ntuple_name)))

# ---- logging helpers ----
class CutLogger:
    def __init__(self, txt_path, csv_path):
        self.txt_path = txt_path
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample","step_idx","step","events","weighted","elapsed_s"])
        # fresh txt header
        with open(txt_path, "a") as f:
            f.write(f"\n==== Cutflow run {RUN_TAG} ====\n")

    def write(self, sample, step_idx, step, events, weighted, elapsed):
        # text
        with open(self.txt_path, "a") as f:
            f.write(f"[{sample:12s}] {step_idx:02d}  {step:30s}  "
                    f"events={events:8d}  weighted={weighted:.6g}  dt={elapsed:.3f}s\n")
        # csv
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([sample, step_idx, step, int(events), f"{weighted:.12g}", f"{elapsed:.6f}"])

logger = CutLogger(TXT_LOG, CSV_LOG)

def log_step(sample, step_idx, step_label, fb, t0):
    nevt = len(fb)
    wsum = weight_sum(fb, sample)
    logger.write(sample, step_idx, step_label, nevt, wsum, time.time() - t0)

def require(mask, name):
    """Utility to guard awkward masks and give readable errors if shapes mismatch."""
    if isinstance(mask, (np.ndarray, ak.Array)) and ak.num(mask, axis=0) is not None:
        return mask
    raise RuntimeError(f"Mask '{name}' has wrong shape/type: {type(mask)}")

# ---- your loop with logging ----
for ntuple_name in ntuple_names:
    start_time = time.time()
    step = 0

    if ntuple_name == 'data23':
        path = "/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/data23_y_BDT_score.root"
        f = uproot.open(path)['nominal']
        fb = f.arrays(variables_data, library="ak")
        fb['VertexBDTScore'] = fb['BDTScore']

        log_step(ntuple_name, step, "loaded", fb, start_time); step += 1

        # ensure photon arrays exist for reweighting usage downstream
        fb = fb[ak.num(fb['ph_eta']) > 0]
        # jet-faking-photon cut (data control)
        mask = (ak.firsts(fb['ph_topoetcone40']) - 2450.)/ak.firsts(fb['ph_pt']) > 0.1
        fb = fb[require(mask, "jetfake")]
        log_step(ntuple_name, step, "jet_faking_photon", fb, start_time); step += 1

        fb = fb[fb['n_ph_baseline'] == 1]
        log_step(ntuple_name, step, "n_ph_baseline==1", fb, start_time); step += 1

    else:
        path = f"/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/mc23d_{ntuple_name}_y_BDT_score.root"
        f = uproot.open(path)['nominal']
        fb = f.arrays(variables, library="ak")

        # add BDT score (same file path, same tree)
        f_BDT = uproot.open(path)['nominal']
        fb_BDT = f_BDT.arrays(["event", "BDTScore"], library="ak")
        if np.all(fb["event"] == fb_BDT["event"]):
            fb["VertexBDTScore"] = fb_BDT["BDTScore"]
        else:
            print(f"[WARN] Event mismatch in {ntuple_name}; BDT not attached")

        log_step(ntuple_name, step, "loaded", fb, start_time); step += 1

        fb = fb[ak.num(fb['ph_eta']) > 0]
        fb = fb[fb['n_ph'] == 1]
        log_step(ntuple_name, step, "n_ph==1", fb, start_time); step += 1

        if ntuple_name in ("Zjets","Wjets"):
            mask = ak.firsts(fb['ph_truth_type']) == 2   # keep e->gamma only
            fb = fb[require(mask, "ph_truth_type==2")]
            log_step(ntuple_name, step, "truth e->gamma", fb, start_time); step += 1

        if ntuple_name == "ggHyyd":
            fb = fb[ak.num(fb['pv_z']) > 0]
            log_step(ntuple_name, step, "pv_z exists", fb, start_time); step += 1
            good_pv = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)
            fb = fb[require(good_pv, "goodPV")]
            log_step(ntuple_name, step, "goodPV", fb, start_time); step += 1

    # --------- BASIC CUTS (shared) ----------
    # NOTE: If 'ggHyyd' is signal without a prompt μ, consider not requiring n_mu==1 for that sample.
    fb = fb[fb['n_mu_baseline'] == 0]
    log_step(ntuple_name, step, "n_mu_baseline==0", fb, start_time); step += 1

    fb = fb[fb['n_el_baseline'] == 0]
    log_step(ntuple_name, step, "n_el_baseline==0", fb, start_time); step += 1

    fb = fb[fb['n_tau_baseline'] == 0]
    log_step(ntuple_name, step, "n_tau_baseline==0", fb, start_time); step += 1

    fb = fb[fb['trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M'] == 1]
    log_step(ntuple_name, step, "trigger==1", fb, start_time); step += 1

    fb = fb[ak.num(fb['ph_pt']) > 0]
    fb = fb[ak.firsts(fb['ph_pt']) >= 50_000]
    log_step(ntuple_name, step, "ph_pt>=50GeV", fb, start_time); step += 1

    fb = fb[fb['met_tst_et'] >= 100_000]
    log_step(ntuple_name, step, "MET>=100GeV", fb, start_time); step += 1

    fb = fb[fb['n_jet_central'] <= 3]
    log_step(ntuple_name, step, "n_jet_central<=3", fb, start_time); step += 1

    mt_tmp = np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) *
                     (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000.0
    fb = fb[mt_tmp >= 100]
    # fb = fb[mt_tmp <= 140]
    log_step(ntuple_name, step, "mT>=100GeV", fb, start_time); step += 1

    fb = fb[fb['VertexBDTScore'] > 0.1]
    log_step(ntuple_name, step, "VertexBDTScore>0.1", fb, start_time); step += 1
    
    metsig_tmp = fb['met_tst_sig'] 
    mask1 = metsig_tmp > 6
    fb = fb[mask1]
    log_step(ntuple_name, step, "met_tst_sig>6", fb, start_time); step += 1
    # mask2 = metsig_tmp <= 13
    # fb = fb[mask1 * mask2]
    
    ph_eta_tmp = np.abs(ak.firsts(fb['ph_eta']))
    fb = fb[ph_eta_tmp < 1.75]
    log_step(ntuple_name, step, "ph_eta<1.75", fb, start_time); step += 1

    dphi_met_phterm_tmp = np.arccos(np.cos(fb['met_tst_phi'] - fb['met_phterm_phi'])) # added cut 3
    fb = fb[dphi_met_phterm_tmp > 1.25]
    log_step(ntuple_name, step, "dphi_met_phterm>1.25", fb, start_time); step += 1

    dmet_tmp = fb['met_tst_noJVT_et'] - fb['met_tst_et']
    mask1 = dmet_tmp > -10000
    fb = fb[mask1]
    log_step(ntuple_name, step, "dmet>-10GeV", fb, start_time); step += 1

    phi1_tmp = ak.firsts(fb['jet_central_phi']) # added cut 7
    phi2_tmp = ak.mask(fb['jet_central_phi'], ak.num(fb['jet_central_phi']) >= 2)[:, 1] 
    dphi_tmp = np.arccos(np.cos(phi1_tmp - phi2_tmp))
    dphi_jj_tmp = ak.fill_none(dphi_tmp, -999)
    fb = fb[dphi_jj_tmp < 2.5]
    log_step(ntuple_name, step, "dphi_jj_central<2.5", fb, start_time); step += 1

    dphi_met_jetterm_tmp = np.where(fb['met_jetterm_et'] != 0,   # added cut 5
                        np.arccos(np.cos(fb['met_tst_phi'] - fb['met_jetterm_phi'])),
                        -999)
    fb = fb[dphi_met_jetterm_tmp <= 0.75]
    log_step(ntuple_name, step, "dphi_met_jetterm<0.75", fb, start_time); step += 1
    

    # ---- sanity check for None ----
    n_none = int(ak.sum(ak.is_none(fb['met_tst_et'])))
    with open(TXT_LOG, "a") as ftxt:
        ftxt.write(f"[{ntuple_name:12s}] None-check met_tst_et: {n_none}\n")

    # optional: free memory
    del fb

print(f"\nLogs written to:\n - {TXT_LOG}\n - {CSV_LOG}\n")