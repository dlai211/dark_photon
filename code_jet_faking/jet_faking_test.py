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

# ---- Memory monitoring helpers ---------------------------------------------
import os, time, gc, math, resource, platform

try:
    import psutil
except Exception:
    psutil = None

def _fmt_bytes(n):
    if n is None: return "n/a"
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024

def _proc_rss():
    """Process RSS (bytes)."""
    if psutil:
        try:
            return psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            pass
    # Fallback: resource on Unix (returns kilobytes)
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1024 if platform.system() != "Darwin" else 1)
    except Exception:
        pass
    # Fallback: /proc/self/statm (pages)
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None

def _sys_mem():
    """(total, available) system memory in bytes if possible."""
    if psutil:
        try:
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:
            pass
    # /proc/meminfo fallback (Linux)
    try:
        kv = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")[0], line.split(":")[1].strip()
                kv[k] = int(v.split()[0]) * 1024  # kB -> B
        total = kv.get("MemTotal")
        avail = kv.get("MemAvailable", kv.get("MemFree"))
        return total, avail
    except Exception:
        return None, None

def _cgroup_mem_limit():
    # cgroup v2
    try:
        p = "/sys/fs/cgroup/memory.max"
        if os.path.exists(p):
            v = open(p).read().strip()
            return None if v in ("max", "") else int(v)
    except Exception:
        pass
    # cgroup v1
    try:
        p = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(p):
            v = open(p).read().strip()
            # Some systems put an enormous number for "no limit"
            lim = int(v)
            if lim >= 1<<60:  # ~exabyte sentinel
                return None
            return lim
    except Exception:
        pass
    return None

class MemoryMonitor:
    def __init__(self, label="run", warn_frac=0.9):
        self.t0 = time.time()
        self.label = label
        self.warn_frac = warn_frac
        self.cgroup_limit = _cgroup_mem_limit()
    def log(self, tag, fb=None):
        rss = _proc_rss()
        total, avail = _sys_mem()
        fb_bytes = ak_nbytes(fb) if fb is not None else None
        lim = self.cgroup_limit or total
        pct = (100.0 * rss / lim) if (rss and lim) else None
        warn = f"  [>{self.warn_frac*100:.0f}% of limit]" if (pct and pct > self.warn_frac*100) else ""
        lim_s = _fmt_bytes(lim) if lim else "n/a"
        pct_s = f"{pct:.1f}%" if pct is not None else "n/a"
        print(f"[mem] {tag:>22s} | RSS={_fmt_bytes(rss)} ({pct_s} of {lim_s})"
              f" | sys_avail={_fmt_bytes(avail)} | fb≈{_fmt_bytes(fb_bytes)}"
              f" | t+{time.time()-self.t0:.1f}s{warn}")
    def collect(self, tag="gc"):
        gc.collect()
        self.log(tag)


def ak_nbytes(arr):
    """Approximate Awkward array memory footprint (bytes) via to_buffers (zero-copy)."""
    try:
        import awkward as ak
        form, length, container = ak.to_buffers(arr)
        return sum(getattr(buf, "nbytes", 0) for buf in container.values())
    except Exception:
        return None
# ---------------------------------------------------------------------------

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


mm = MemoryMonitor(label="jet-faking", warn_frac=0.90)  ### NEW
mm.log("start")                                         ### NEW

tot = []
data = pd.DataFrame()
unweighted_bcut, weighted_bcut, unweighted_acut, weighted_acut = [], [], [], []
ntuple_names = ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct', 'data23']

for i in range(len(ntuple_names)):
    ucut, wcut = [], []
    start_time = time.time()
    ntuple_name = ntuple_names[i]

    if ntuple_name == 'data23':  # data
        path = f"/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/data23_y_BDT_score.root"
        print('processing file: ', path)
        mm.log("open data")                          ### NEW
        f = uproot.open(path)['nominal']
        fb = f.arrays(variables_data, library="ak")
        mm.log("loaded data", fb)                    ### NEW

        fb = fb[ak.num(fb['ph_eta']) > 0]
        mm.log("has photon", fb)                     ### NEW

        mask1 = (ak.firsts(fb['ph_topoetcone40'])-2450.)/ak.firsts(fb['ph_pt']) > 0.1
        fb = fb[mask1]
        fb = fb[fb['n_ph_baseline'] == 1]
        mm.log("data preselect", fb)                 ### NEW

    else:  # MC
        path = f"/data/tmathew/ntups/mc23d/{ntuple_name}_y.root"
        path_BDT = f"/data/fpiazza/ggHyyd/Ntuples/MC23d/withVertexBDT/mc23d_{ntuple_name}_y_BDT_score.root"
        print('processing file: ', path)

        mm.log("open mc")                            ### NEW
        f = uproot.open(path)['nominal']
        fb = f.arrays(variables, library="ak")
        mm.log("mc loaded", fb)                      ### NEW

        # add BDT score
        f_BDT = uproot.open(path_BDT)['nominal']
        fb_BDT = f_BDT.arrays(["event", "BDTScore"], library="ak")
        mm.log("BDT loaded", fb_BDT)                 ### NEW

        tmp = fb["event"] == fb_BDT["event"]
        if np.all(tmp) == True:
            fb["BDTScore"] = fb_BDT["BDTScore"]
        else:
            print("Something is wrong, need arranging")
        mm.log("BDT merged", fb)                     ### NEW

        fb = fb[ak.num(fb['ph_eta']) > 0]
        fb = fb[fb['n_ph'] == 1]
        mm.log("photon select", fb)                  ### NEW

        if ntuple_name in ('Zjets', 'Wjets'):
            mask = ak.firsts(fb['ph_truth_type']) == 2
            fb = fb[mask]
            mm.log("truth e->gamma", fb)             ### NEW

        if ntuple_name == 'ggHyyd':
            fb = fb[ak.num(fb['pv_z']) > 0]
            good_pv_tmp = (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5)
            fb = fb[good_pv_tmp]
            mm.log("goodPV", fb)                     ### NEW

    print_cut(ntuple_name, fb, 'before cut')
    mm.log("before basic cuts", fb)                  ### NEW

    fb = fb[fb['n_mu_baseline'] == 0]
    fb = fb[fb['n_el_baseline'] == 0]
    fb = fb[fb['n_tau_baseline'] == 0]
    fb = fb[fb['trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M']==1]
    fb = fb[ak.num(fb['ph_pt']) > 0]
    fb = fb[ak.firsts(fb['ph_pt']) >= 50000]
    fb = fb[fb['met_tst_et'] >= 100000]
    fb = fb[fb['n_jet_central'] <= 4]
    mm.log("after basic cuts", fb)                   ### NEW

    mt_tmp = np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) *
                     (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000
    mask1 = mt_tmp >= 100
    fb = fb[mask1]
    mm.log("after mT cut", fb)                       ### NEW

    print_cut(ntuple_name, fb, 'after basic cut')

    # optional: force free temp arrays and run GC
    del mt_tmp, mask1
    mm.collect("gc after cuts")                      ### NEW

    tot.append(fb)
    mm.log("appended to tot", fb)                    ### NEW

    # aggressively release intermediates
    del fb, f
    if ntuple_name != 'data23':
        del f_BDT, fb_BDT, tmp
    mm.collect("gc end of sample")                   ### NEW

    print(f"Reading Time for {ntuple_name}: {(time.time()-start_time):.1f} s\n")

mm.log("done all samples")                           ### NEW
