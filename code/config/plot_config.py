import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.special import betainc

def getWeight(fb, sample):
    # reweighting for jet faking photons
    if sample == "data23_y":
        abs_eta = abs(ak.firsts(fb['ph_eta'])) # leading photon per event
        sf = ak.full_like(abs_eta, 0.0)

        sf = ak.where((abs_eta > 0.0) & (abs_eta <= 0.6), 1.841, sf)
        sf = ak.where((abs_eta > 0.6) & (abs_eta <= 1.37), 2.102, sf)
        sf = ak.where((abs_eta > 1.37) & (abs_eta <= 1.52), 1.875, sf)
        sf = ak.where((abs_eta > 1.52) & (abs_eta <= 1.81), 1.911, sf)
        sf = ak.where((abs_eta > 1.81) & (abs_eta <= 2.37), 2.382, sf)
        return sf

    if sample == "data24_y":
        abs_eta = abs(ak.firsts(fb['ph_eta']))
        sf = ak.full_like(abs_eta, 0.0)

        sf = ak.where((abs_eta > 0.0) & (abs_eta <= 0.6), 1.865, sf)
        sf = ak.where((abs_eta > 0.6) & (abs_eta <= 1.37), 1.453, sf)
        sf = ak.where((abs_eta > 1.37) & (abs_eta <= 1.52), 1.957, sf)
        sf = ak.where((abs_eta > 1.52) & (abs_eta <= 1.81), 1.894, sf)
        sf = ak.where((abs_eta > 1.81) & (abs_eta <= 2.37), 1.605, sf)
        return sf

    # reweighting for electron faking photons
    if sample == "data23_eprobe":
        el_pt  = ak.fill_none(ak.firsts(fb["el_pt"]),  -1)   # MeV
        abs_eta = ak.fill_none(abs(ak.firsts(fb["el_eta"])), 99.0)
    
        scale = (
            (1.257 * ((el_pt > 50000)   & (el_pt <= 52000))) +
            (1.040 * ((el_pt > 52000)   & (el_pt <= 54000))) +
            (1.104 * ((el_pt > 54000)   & (el_pt <= 56000))) +
            (1.064 * ((el_pt > 56000)   & (el_pt <= 58000))) +
            (1.079 * ((el_pt > 58000)   & (el_pt <= 60000))) +
            (1.113 * ((el_pt > 60000)   & (el_pt <= 62000))) +
            (1.088 * ((el_pt > 62000)   & (el_pt <= 65000))) +
            (1.077 * ((el_pt > 65000)   & (el_pt <= 70000))) +
            (1.074 * ((el_pt > 70000)   & (el_pt <= 80000))) +
            (1.099 * ((el_pt > 80000)   & (el_pt <= 100000))) +
            (1.097 * ((el_pt > 100000)  & (el_pt <= 200000)))
        )
    
        norm = (
            (0.02695 * ((abs_eta > 0.00) & (abs_eta <= 0.60))) + 
            (0.02822 * ((abs_eta > 0.60) & (abs_eta <= 1.37))) +
            (0.00000 * ((abs_eta > 1.37) & (abs_eta <= 1.52))) +
            (0.05412 * ((abs_eta > 1.52) & (abs_eta <= 1.81))) +
            (0.10188 * ((abs_eta > 1.81) & (abs_eta <= 2.37)))
        )
        
        return scale * norm

    if sample == "data24_eprobe":
        el_pt  = ak.fill_none(ak.firsts(fb["el_pt"]),  -1)   # MeV
        abs_eta = ak.fill_none(abs(ak.firsts(fb["el_eta"])), 99.0)
    
        scale = (
            (1.318 * ((el_pt > 50000)   & (el_pt <= 52000))) +
            (1.103 * ((el_pt > 52000)   & (el_pt <= 54000))) +
            (1.118 * ((el_pt > 54000)   & (el_pt <= 56000))) +
            (1.134 * ((el_pt > 56000)   & (el_pt <= 58000))) +
            (1.107 * ((el_pt > 58000)   & (el_pt <= 60000))) +
            (1.129 * ((el_pt > 60000)   & (el_pt <= 62000))) +
            (1.137 * ((el_pt > 62000)   & (el_pt <= 65000))) +
            (1.105 * ((el_pt > 65000)   & (el_pt <= 70000))) +
            (1.125 * ((el_pt > 70000)   & (el_pt <= 80000))) +
            (1.123 * ((el_pt > 80000)   & (el_pt <= 100000))) +
            (1.109 * ((el_pt > 100000)  & (el_pt <= 200000)))
        )
    
        norm = (
            (0.03734 * ((abs_eta > 0.00) & (abs_eta <= 0.60))) + 
            (0.02868 * ((abs_eta > 0.60) & (abs_eta <= 1.37))) +
            (0.00000 * ((abs_eta > 1.37) & (abs_eta <= 1.52))) +
            (0.04860 * ((abs_eta > 1.52) & (abs_eta <= 1.81))) +
            (0.09252 * ((abs_eta > 1.81) & (abs_eta <= 2.37)))
        )
        
        return scale * norm

    if sample.startswith("mc23d"):
        lumi = 36000
    if sample.startswith("mc23e"):
        lumi = 109000
    weight = fb['mconly_weight']/fb['mc_weight_sum']*fb['xsec_ami']*fb['filter_eff_ami']*fb['kfactor_ami']*fb['pu_weight']*fb['jvt_weight']*1000*lumi

    if any(signal in sample for signal in ["ggHyyd", "WH", "VBF", "ZH"]):
        xsec_sig = 0.052 #if ( period == 'Run3' or 'mc23' in period ) else 0.048
        # if sample != 'ggHyyd' : xsec_sig = fb['xsec_ami']
        br = 0.01
        weight = fb['mconly_weight']/fb['mc_weight_sum']*xsec_sig*fb['pu_weight']*fb['jvt_weight']*fb['filter_eff_ami']*fb['kfactor_ami']*1000*lumi*br

    return weight

def zbi(s, b, sigma_b_frac=0.3):
    if b <= 0:
        return 0.0
    tau = 1.0 / (b * sigma_b_frac * sigma_b_frac)
    n_on = s + b
    n_off = b * tau

    # probability
    P_Bi = betainc(n_on, n_off + 1, 1.0 / (1.0 + tau))

    if P_Bi <= 0:
        return 0.0

    # finally, ZBi is quantile (inverse CDF of normal) at 1 - P_Bi
    Z_Bi = norm.ppf(1.0 - P_Bi)

    return Z_Bi

def getSampleDict():
    sample_dict = {
        # --- Signal ---
        'ggHyyd': {
            'color': 'red',
            'legend': r'ggH, H$\rightarrow\gamma\gamma_{d}$',
            'tree': 'nominal',
            'filenames': ['mc23d_ggHyyd_y'],
        },

        # --- MC backgrounds (combined mc23d + mc23e) ---
        'Zgamma': {
            'color': '#e6550d',
            'legend': r'Z($\nu\nu$)+$\gamma$',
            'tree': 'nominal',
            'filenames': ['mc23d_Zgamma_y', 'mc23e_Zgamma_y'],
        },
        'Wgamma': {
            'color': 'darkorange',
            'legend': r'W($\ell\nu$)+$\gamma$',
            'tree': 'nominal',
            'filenames': ['mc23d_Wgamma_y', 'mc23e_Wgamma_y'],
        },
        'gammajet_direct': {
            'color': 'royalblue',
            'legend': r'$\gamma$+jets direct',
            'tree': 'gammajets',
            'filenames': ['mc23d_gammajet_direct_y', 'mc23e_gammajet_direct_y'],
        },

        # --- DATA (combined 2023 + 2024) ---
        'data_y': {
            'color': 'deepskyblue',
            'legend': r'data $j\to\gamma$',
            'tree': 'nominal',
            'filenames': ['data23_y', 'data24_y'],
        },
        'data_eprobe': {
            'color': 'turquoise',
            'legend': r'data $e\to\gamma$',
            'tree': 'nominal',
            'filenames': ['data23_eprobe', 'data24_eprobe'],
        },
    }
    return sample_dict
sample_dict = getSampleDict()

def getVarDict(fb, process, var_name=None):
    var_dict = {}

    if var_name is None or var_name == 'n_ph':
        var_dict['n_ph'] = {
            'var': fb['n_ph'],
            'bins': np.linspace(0, 7, 7+1),
            'title': r'$N_{ph}$'
        }

    if var_name is None or var_name == 'n_ph_baseline':
        var_dict['n_ph_baseline'] = {
            'var': fb['n_ph_baseline'],
            'bins': np.linspace(0, 7, 7+1),
            'title': r'$N_{ph\_baseline}$'
        }

    if var_name is None or var_name == 'n_el':
        var_dict['n_el'] = {
            'var': fb['n_el'],
            'bins': np.linspace(0, 7, 7+1),
            'title': r'$N_{el}$'
        }

    if var_name is None or var_name == 'n_el_baseline':
        var_dict['n_el_baseline'] = {
            'var': fb['n_el_baseline'],
            'bins': np.linspace(0, 7, 7+1),
            'title': r'$N_{el\_baseline}$'
        }

    if var_name is None or var_name == 'n_mu_baseline':
        var_dict['n_mu_baseline'] = {
            'var': fb['n_mu_baseline'],
            'bins': np.linspace(0, 7, 7+1),
            'title': r'$N_{mu\_baseline}$'
        }

    if var_name is None or var_name == 'n_tau_baseline':
        var_dict['n_tau_baseline'] = {
            'var': fb['n_tau_baseline'],
            'bins': np.linspace(0, 7, 7+1),
            'title': r'$N_{tau\_baseline}$'
        }

    if var_name is None or var_name == 'mt':
        var_dict['mt'] = {
            'var': np.sqrt(2 * fb['met_tst_et'] * ak.firsts(fb['ph_pt']) * 
                           (1 - np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi'])))) / 1000,
            'bins': np.linspace(0, 300, 15+1),
            'title': r'$m_T\ [GeV]$',
            'shift': '+0'
        }

    if var_name is None or var_name == 'metsig':
        var_dict['metsig'] = {
            'var': fb['met_tst_sig'],
            'bins': np.linspace(0, 30, 15+1),
            'title': r'$E_T^{miss}\ significance$',
            'shift': '*1'
        }

    if var_name is None or var_name == 'metsigres':
        var_dict['metsigres'] = {
            'var': fb['met_tst_et'] / fb['met_tst_sig'],
            'bins': np.linspace(0, 100000, 50+1),
            'title': r'$E_T^{miss}\ significance$',
            'shift': '*1'
        }

    if var_name is None or var_name == 'met':
        var_dict['met'] = {
            'var': fb['met_tst_et'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ [MeV]$',
            'shift': '+50000'
        }

    if var_name is None or var_name == 'met_noJVT':
        var_dict['met_noJVT'] = {
            'var': fb['met_tst_noJVT_et'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ [MeV]$'
        }

    if var_name is None or var_name == 'dmet':
        var_dict['dmet'] = {
            'var': fb['dmet'],
            'bins': np.linspace(-100000, 100000, 20+1),
            'title': r'$E_{T,\mathrm{noJVT}}^{miss}-E_T^{miss}\ [MeV]$',
            'shift': '*1'
        }

    if var_name is None or var_name == 'ph_pt':
        var_dict['ph_pt'] = {
            'var': ak.firsts(fb['ph_pt']),
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{\gamma}\;\mathrm{or}\;p_T^{e}\;(e\to\gamma)\;[{\rm MeV}]$',
            'shift': '-150000'
        }

    if var_name is None or var_name == 'ph_eta':
        var_dict['ph_eta'] = {
            'var': np.abs(ak.firsts(fb['ph_eta'])),
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\eta^{\gamma}\;\mathrm{or}\;\eta^{e}\;(e\to\gamma)$'
        }

    if var_name is None or var_name == 'ph_phi':
        var_dict['ph_phi'] = {
            'var': ak.firsts(fb['ph_phi']),
            'bins': np.linspace(-4, 4, 50+1),
            'title': r'$\phi^{\gamma}\;\mathrm{or}\;\phi^{e}\;(e\to\gamma)$'
        }

    if var_name is None or var_name == 'pv_ntracks':
        var_dict['pv_ntracks'] = {
            'var': ak.firsts(fb['pv_ntracks']),
            'bins': np.linspace(0, 200, 50+1),
            'title': 'pv_ntracks'
        }
        
    if var_name is None or var_name == "jet_central_eta":
        jet_central_eta_tmp = ak.firsts(fb['jet_central_eta'])
        var_dict['jet_central_eta'] = {
            'var': ak.fill_none(jet_central_eta_tmp, -999),
            'bins': np.linspace(-4, 4, 50+1), 
            'title': r'$\eta^{\mathrm{jets}}$'
        }

    if var_name is None or var_name == "jet_central_vecSumPt":
        var_dict['jet_central_vecSumPt'] = {
            'var': ak.fill_none(fb['jet_central_vecSumPt'], -999),
            'bins': np.linspace(0, 300000, 50+1), 
            'title': r'$\vec{\sum}p_T^{\mathrm{jet,\,central}}\ [\mathrm{MeV}]$'
        }

    # Jet central pt1 (first jet)
    if var_name is None or var_name == "jet_central_pt1":
        jet_central_pt1_tmp = ak.firsts(fb['jet_central_pt'])
        var_dict['jet_central_pt1'] = {
            'var': ak.fill_none(jet_central_pt1_tmp, -999),
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{j1}\ [MeV]$'
        }

    # Jet central pt2 (second jet, if available)
    if var_name is None or var_name == "jet_central_pt2":
        jet_central_pt2_tmp = ak.mask(fb['jet_central_pt'], ak.num(fb['jet_central_pt']) >= 2)[:, 1]
        var_dict['jet_central_pt2'] = {
            'var': ak.fill_none(jet_central_pt2_tmp, -999),
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{j2}\ [MeV]$'
        }

    if var_name is None or var_name == 'dphi_met_phterm':
        var_dict['dphi_met_phterm'] = {
            'var': fb['dphi_met_phterm'],
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})$',
        }

    if var_name is None or var_name == 'dphi_met_jetterm':
        var_dict['dphi_met_jetterm'] = {
            'var': fb['dphi_met_jetterm'],
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{jet})\;mathrm{or}\;\Delta\phi(E_T^{miss},\, E_T^{e})\; (e\to\gamma)$'
        }

    if var_name is None or var_name == 'dphi_phterm_jetterm':
        dphi_met_jetterm = fb['dphi_met_jetterm']
        dphi_met_phterm = fb['dphi_met_phterm']
        var_dict['dphi_phterm_jetterm'] = {
            'var': np.arctan2(np.sin(dphi_met_jetterm - dphi_met_phterm),
                              np.cos(dphi_met_jetterm - dphi_met_phterm)),
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(E_T^{\gamma},\, E_T^{jet})\;mathrm{or}\;\Delta\phi(E_T^{e},\, E_T^{jet})\; (e\to\gamma)$'
        }

    # # Fail JVT jet pt1 (first element)
    if var_name is None or var_name == 'failJVT_jet_pt1':
        failJVT_jet_pt_tmp = ak.firsts(fb['failJVT_jet_pt'])
        var_dict['failJVT_jet_pt1'] = {
            'var': ak.fill_none(failJVT_jet_pt_tmp, -999),
            'bins': np.linspace(20000, 60000, 40+1),
            'title': r'$p_T^{\mathrm{noJVT\ jet1}}\ [MeV]$'
        }

    
    if var_name is None or var_name == 'failJVT_jet_vecSumPt':
        var_dict['failJVT_jet_vecSumPt'] = {
            'var': ak.fill_none(fb['failJVT_jet_vecSumPt'], -999),
            'bins': np.linspace(0, 100000, 50+1),
            'title': r'$\vec{\sum}p_T^{\mathrm{jet,\,failJVT}}\ [\mathrm{MeV}]$'
        }

    if var_name is None or var_name == 'softerm':
        var_dict['softerm'] = {
            'var': fb['met_softerm_tst_et'],
            'bins': np.linspace(0, 100000, 50+1),
            'title': r'$E_T^{soft}\ [MeV]$'
        }

    if var_name is None or var_name == 'jetterm':
        var_dict['jetterm'] = {
            'var': fb['met_jetterm_et'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{jet}\ [MeV]$'
        }

    if var_name is None or var_name == 'jetterm_sumet':
        var_dict['jetterm_sumet'] = {
            'var': fb['met_jetterm_sumet'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{jet}\ [MeV]$'
        }

    if var_name is None or var_name == 'n_jet':
        var_dict['n_jet'] = {
            'var': fb['n_jet'],
            'bins': np.linspace(0, 10, 10+1),
            'title': r'$N_{jet}$'
        }

    if var_name is None or var_name == 'n_jet_central':
        var_dict['n_jet_central'] = {
            'var': fb['n_jet_central'],
            'bins': np.linspace(0, 10, 10+1),
            'title': r'$N_{jet}^{central}$'
        }

    if var_name is None or var_name == 'n_jet_fwd':
        var_dict['n_jet_fwd'] = {
            'var': fb['n_jet'] - fb['n_jet_central'],
            'bins': np.linspace(0, 10, 10+1),
            'title': r'$N_{jet}^{fwd}$'
        }

    if var_name is None or var_name == 'central_jets_fraction':
        var_dict['central_jets_fraction'] = {
            'var': np.where(fb['n_jet'] > 0, fb['n_jet_central']/fb['n_jet'], -1),
            'bins': np.linspace(-1, 2, 50+1),
            'title': r'Central jets fraction'
        }

    # Balance: (met_tst_et+ph_pt[0]) divided by the sum over jet_central_pt.
    if var_name is None or var_name == 'balance':
        # balance_tmp = fb["central_balance"]
        # cond = ak.fill_none(balance_tmp == -10, False)
        # balance = ak.where(cond, -999, balance_tmp)
        
        sumet_tmp = fb['jet_central_vecSumPt']
        expr = (fb['met_tst_et'] + ak.firsts(fb['ph_pt'])) / ak.where(sumet_tmp != 0, sumet_tmp, 1)
        balance = ak.where(sumet_tmp != 0, expr, -10) 

        var_dict['balance'] = {
            'var': balance,
            'bins': np.linspace(-1, 15, 50+1),
            'title': r'balance'
        }


    # dphi_jj: Use Alt$ logic – if jet_central_phi has at least two entries, compute the difference; else -1.
    # Here we use a Python conditional (this assumes fb['jet_central_phi'] is an array with shape information).
    if var_name is None or var_name == 'dphi_jj':
        dphi_jj_tmp = fb['dphi_central_jj']
        dphi_jj_tmp = ak.where(dphi_jj_tmp == -10, np.nan, dphi_jj_tmp)
        dphi_jj_tmp = np.arccos(np.cos(dphi_jj_tmp))
        dphi_jj_tmp = ak.where(np.isnan(dphi_jj_tmp), -999, dphi_jj_tmp)
        var_dict['dphi_jj'] = {
            'var': dphi_jj_tmp,
            'bins': np.linspace(-1, 4, 20+1),
            'title': r'$\Delta\phi(j1,\, j2)$'
        }
    
    if var_name is None or var_name == 'VertexBDTScore':
        var_dict['VertexBDTScore'] = {
            'var': fb['VertexBDTScore'],
            'bins': np.arange(0, 1+0.11, 0.11),
            'title': 'VertexBDTScore'
        }
    
    return var_dict