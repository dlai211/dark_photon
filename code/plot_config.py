import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.special import betainc

def getWeight(fb, sample, jet_faking=False, electron_faking=False):
    # lumi = 25767.5
    lumi = 135000
    weight = fb['mconly_weight']/fb['mc_weight_sum']*fb['xsec_ami']*fb['filter_eff_ami']*fb['kfactor_ami']*fb['pu_weight']*fb['jvt_weight']*1000*lumi
    if sample in ['ggHyyd','WH','VBF','ZH'] : 
        xsec_sig = 0.052 #if ( period == 'Run3' or 'mc23' in period ) else 0.048
        # if sample != 'ggHyyd' : xsec_sig = fb['xsec_ami']
        br = 0.01
        weight = fb['mconly_weight']/fb['mc_weight_sum']*xsec_sig*fb['pu_weight']*fb['jvt_weight']*fb['filter_eff_ami']*fb['kfactor_ami']*1000*lumi*br

    # reweighting for data-driven 
    if jet_faking:
        abs_eta = ak.abs(ak.firsts(fb['ph_eta'])) # leading photon per event
        sf = ak.full_like(abs_eta, 0.0)

        sf = ak.where((abs_eta > 0.0) & (abs_eta <= 0.6), 1.84, sf)
        sf = ak.where((abs_eta > 0.6) & (abs_eta <= 1.37), 2.14, sf)
        sf = ak.where((abs_eta > 1.37) & (abs_eta <= 1.52), 0.0, sf)
        sf = ak.where((abs_eta > 1.52) & (abs_eta <= 1.81), 1.99, sf)
        sf = ak.where((abs_eta > 1.81) & (abs_eta <= 2.37), 2.21, sf)
        return sf

    if electron_faking:
        el_pt_GeV = ak.firsts(fb['el_pt']) * 0.001  # leading electron pt in GeV
        abs_eta = abs(ak.firsts(fb['el_eta']))   # leading electron |eta|

        # pt-dependent scale factor
        scale = ak.full_like(el_pt_GeV, 1.0)

        scale = ak.where((el_pt_GeV > 50) & (el_pt_GeV <= 52), 1.08, scale)
        scale = ak.where((el_pt_GeV > 52) & (el_pt_GeV <= 54), 1.22, scale)
        scale = ak.where((el_pt_GeV > 54) & (el_pt_GeV <= 56), 1.14, scale)
        scale = ak.where((el_pt_GeV > 56) & (el_pt_GeV <= 58), 1.11, scale)
        scale = ak.where((el_pt_GeV > 58) & (el_pt_GeV <= 60), 1.13, scale)
        scale = ak.where((el_pt_GeV > 60) & (el_pt_GeV <= 62), 1.16, scale)
        scale = ak.where((el_pt_GeV > 62) & (el_pt_GeV <= 65), 1.12, scale)
        scale = ak.where((el_pt_GeV > 65) & (el_pt_GeV <= 70), 1.10, scale)
        scale = ak.where((el_pt_GeV > 70) & (el_pt_GeV <= 80), 1.05, scale)
        scale = ak.where((el_pt_GeV > 80) & (el_pt_GeV <= 100), 1.09, scale)
        scale = ak.where((el_pt_GeV > 100) & (el_pt_GeV <= 200), 1.02, scale)

        # eta-bin normalization factors
        norm = ak.full_like(abs_eta, 0.0)

        norm = ak.where((abs_eta > 0.0) & (abs_eta <= 0.6), 0.02808, norm)
        norm = ak.where((abs_eta > 0.6) & (abs_eta <= 1.37), 0.02975, norm)
        norm = ak.where((abs_eta > 1.37) & (abs_eta <= 1.52), 0.0, norm)  # crack region
        norm = ak.where((abs_eta > 1.52) & (abs_eta <= 1.81), 0.05631, norm)
        norm = ak.where((abs_eta > 1.81) & (abs_eta <= 2.37), 0.09524, norm)

        return scale * norm
        

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
    sample_dict = {}
    sample_dict['Zjets'] = {
        'color': 'darkgreen',   # approximates ROOT.kGreen-2
        'legend': r'Z($\nu\nu$, ll)+jets',
        'tree': 'nominal',
        'filenames': ['Zjets']
    }
    sample_dict['Zgamma'] = {
        'color': '#e6550d',      # approximates ROOT.kOrange+7
        'legend': r'Z($\nu\nu$)+$\gamma$',
        'tree': 'nominal',
        'filenames': ['Zgamma']
    }
    sample_dict['Wgamma'] = {
        'color': 'darkorange',  # approximates ROOT.kOrange+1
        'legend': r'W($l\nu$)+$\gamma$',
        'tree': 'nominal',
        'filenames': ['Wgamma']
    }
    sample_dict['Wjets'] = {
        'color': 'teal',        # approximates ROOT.kTeal+5
        'legend': r'W($l\nu$)+jets',
        'tree': 'nominal',
        'filenames': ['Wjets']
    }
    sample_dict['gammajet_direct'] = {
        'color': 'royalblue',   # approximates ROOT.kBlue+2
        'legend': r'$\gamma$+jets direct',
        'tree': 'gammajets',
        'filenames': ['gammajet_direct']
    }
    sample_dict['gammajet_frag'] = {
        'color': 'navy',        # approximates ROOT.kBlue-5
        'legend': r'$\gamma$+jets frag',
        'tree': 'gammajets',
        'filenames': ['gammajet_frag']
    }
    sample_dict['dijet'] = {
        'color': 'cyan',        # approximates ROOT.kCyan+1
        'legend': 'multijets',
        'tree': 'dijets',
        'filenames': ['dijet']
    }
    sample_dict['ggHyyd'] = {
        'color': 'red',         # approximates ROOT.kRed
        'legend': r'ggH, H$\rightarrow\gamma\gamma_{d}$',
        'tree': 'nominal',
        'filenames': ['ggHyyd']
    }
    return sample_dict
sample_dict = getSampleDict()

def getVarDict(fb, process, var_name=None):
    var_dict = {}

    # this has the same size as weight, so don't need adjustment on weighting
    if var_name is None or var_name == 'vtx_sumPt':
        var_dict['vtx_sumPt'] = {
            'var': ak.flatten(fb['vtx_sumPt']),
            'bins': np.linspace(0, 100, 20+1),  # 21 edges for 20 bins
            'title': r'vtx\_sumPt'
        }

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

    if var_name is None or var_name == 'puWeight':
        var_dict['puWeight'] = {
            'var': fb['pu_weight'],
            'bins': np.linspace(0, 2, 50+1),
            'title': r'PU weight',
            'shift': '+0'
        }

    if var_name is None or var_name == 'actualIntPerXing':
        var_dict['actualIntPerXing'] = {
            'var': fb['actualIntPerXing'],
            'bins': np.linspace(0, 100, 50+1),
            'title': r'$\langle\mu\rangle$',
            'shift': '+0'
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

    if var_name is None or var_name == 'met_cst':
        var_dict['met_cst'] = {
            'var': fb['met_cst_et'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ CST\ [MeV]$'
        }

    if var_name is None or var_name == 'met_track':
        var_dict['met_track'] = {
            'var': fb['met_track_et'],
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ Track\ [MeV]$'
        }

    if var_name is None or var_name == 'dmet':
        var_dict['dmet'] = {
            'var': fb['met_tst_noJVT_et'] - fb['met_tst_et'],
            'bins': np.linspace(-100000, 100000, 20+1),
            'title': r'$E_{T,\mathrm{noJVT}}^{miss}-E_T^{miss}\ [MeV]$',
            'shift': '*1'
        }

    if var_name is None or var_name == 'ph_pt':
        var_dict['ph_pt'] = {
            'var': ak.firsts(fb['ph_pt']),
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{\gamma}\ [MeV]$',
            'shift': '-150000'
        }

    if var_name is None or var_name == 'ph_eta':
        var_dict['ph_eta'] = {
            'var': np.abs(ak.firsts(fb['ph_eta'])),
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\eta^{\gamma}$'
        }

    if var_name is None or var_name == 'ph_phi':
        var_dict['ph_phi'] = {
            'var': ak.firsts(fb['ph_phi']),
            'bins': np.linspace(-4, 4, 50+1),
            'title': r'$\phi^{\gamma}$'
        }

    if var_name is None or var_name == "jet_central_eta":
        jet_central_eta_tmp = ak.firsts(fb['jet_central_eta'])
        var_dict['jet_central_eta'] = {
            'var': ak.fill_none(jet_central_eta_tmp, -999),
            'bins': np.linspace(-4, 4, 50+1), 
            'title': r'$\eta^{\mathrm{jets}}$'
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

    # Jet central pt (all jets)
    if var_name is None or var_name == "jet_central_pt":
        weight_tmp = getWeight(fb, process)
        expanded_weights = ak.flatten(ak.broadcast_arrays(weight_tmp, fb['jet_central_pt'])[0])
        var_dict['jet_central_pt'] = {
            'var': ak.flatten(fb['jet_central_pt']),
            'weight': expanded_weights,
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{j}\ [MeV]$'
    }

    if var_name is None or var_name == 'dphi_met_phterm':
        var_dict['dphi_met_phterm'] = {
            'var': np.arccos(np.cos(fb['met_tst_phi'] - fb['met_phterm_phi'])),
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})$',
            'shift': '+0'
        }

    if var_name is None or var_name == 'dphi_met_ph':
        var_dict['dphi_met_ph'] = {
            'var': np.arccos(np.cos(fb['met_tst_phi'] - ak.firsts(fb['ph_phi']))),
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})$'
        }

    if var_name is None or var_name == 'dphi_met_jetterm':
        var_dict['dphi_met_jetterm'] = {
            'var': np.where(fb['met_jetterm_et'] != 0,
                            np.arccos(np.cos(fb['met_tst_phi'] - fb['met_jetterm_phi'])),
                            -999),
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{jet})$'
        }

    if var_name is None or var_name == 'dphi_phterm_jetterm':
        var_dict['dphi_phterm_jetterm'] = {
            'var': np.where(fb['met_jetterm_et'] > 0,
                            np.arccos(np.cos(fb['met_phterm_phi'] - fb['met_jetterm_phi'])),
                            -999),
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(E_T^{\gamma},\, E_T^{jet})$'
        }

    if var_name is None or var_name == 'dphi_met_phterm_minus_dphi_met_jetterm':
        var_dict['dphi_met_phterm_minus_dphi_met_jetterm'] = {
            'var': np.where(fb['met_jetterm_et'] > 0,
                            np.arccos(np.cos(fb['met_tst_phi'] - fb['met_phterm_phi'])) -
                            np.arccos(np.cos(fb['met_tst_phi'] - fb['met_jetterm_phi'])),
                            -999),  
            'bins': np.linspace(-4, 4, 100+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})-\Delta\phi(E_T^{miss},\, E_T^{jet})$'
        }

    if var_name is None or var_name == 'dphi_met_phterm_divide_dphi_met_jetterm':
        numerator_tmp = np.arccos(np.cos(fb['met_tst_phi'] - fb['met_phterm_phi']))
        denominator_tmp = np.arccos(np.cos(fb['met_tst_phi'] - fb['met_jetterm_phi']))
        var_dict['dphi_met_phterm_divide_dphi_met_jetterm'] = {
            'var': np.where((fb['met_jetterm_et'] > 0) & (denominator_tmp != 0),
                            numerator_tmp / denominator_tmp,
                            -999),  
            'bins': np.linspace(-4, 4, 100+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})/\Delta\phi(E_T^{miss},\, E_T^{jet})$'
        }


    # Delta phi (photon vs. central jet1)
    if var_name is None or var_name == 'dphi_ph_centraljet1':
        dphi_ph_centraljet1_tmp = np.arccos(np.cos(ak.firsts(fb['ph_phi']) - ak.firsts(fb['jet_central_phi'])))
        var_dict['dphi_ph_centraljet1'] = {
            'var': ak.fill_none(dphi_ph_centraljet1_tmp, -999),
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(\gamma,\, j1)$'
        }

    # # Delta phi (photon vs. jet1)
    if var_name is None or var_name == 'dphi_ph_jet1':
        dphi_ph_jet1_tmp = np.arccos(np.cos(ak.firsts(fb['ph_phi']) - ak.firsts(fb['jet_central_phi'])))
        var_dict['dphi_ph_jet1'] = {
            'var': ak.fill_none(dphi_ph_jet1_tmp, -999),
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(\gamma,\, j1)$'
        }

    # # Delta phi (central jet1 vs. jet2) (repeated with dphi_jj)
    # if var_name is None or var_name == 'dphi_central_jet1_jet2':
    #     phi1_tmp = ak.firsts(fb['jet_central_phi'])
    #     phi2_tmp = ak.mask(fb['jet_central_phi'], ak.num(fb['jet_central_phi']) >= 2)[:, 1]
    #     dphi_central_tmp = np.arccos(np.cos(phi1_tmp - phi2_tmp))
    #     var_dict['dphi_central_jet1_jet2'] = {
    #         'var': ak.fill_none(dphi_central_tmp, -999),
    #         'bins': np.linspace(0, 4, 50+1),
    #         'title': r'$\Delta\phi(j1,\, j2)$'
    #     }

    # Met plus photon pt
    if var_name is None or var_name == 'metplusph':
        var_dict['metplusph'] = {
            'var': fb['met_tst_et'] + ak.firsts(fb['ph_pt']),
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}+p_T^{\gamma}\ [MeV]$'
        }

    # # Fail JVT jet pt (all)
    if var_name is None or var_name == 'failJVT_jet_pt':
        weight_tmp = getWeight(fb, process)
        expanded_weights = ak.flatten(ak.broadcast_arrays(weight_tmp, fb['failJVT_jet_pt'])[0])
        var_dict['failJVT_jet_pt'] = {
            'var': ak.flatten(fb['failJVT_jet_pt']),
            'weight': expanded_weights,
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{\mathrm{noJVT\ jet}}\ [MeV]$'
        }

    # # Fail JVT jet pt1 (first element)
    if var_name is None or var_name == 'failJVT_jet_pt1':
        failJVT_jet_pt_tmp = ak.firsts(fb['failJVT_jet_pt'])
        var_dict['failJVT_jet_pt1'] = {
            'var': ak.fill_none(failJVT_jet_pt_tmp, -999),
            'bins': np.linspace(20000, 60000, 40+1),
            'title': r'$p_T^{\mathrm{noJVT\ jet1}}\ [MeV]$'
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

    # if var_name is None or var_name == 'vertex':
    #     var_dict['vertex'] = {
    #         'var': (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) == 
    #                 np.min(np.abs(ak.firsts(fb['pv_truth_z']) - fb['pv_z']))),
    #         'bins': np.linspace(0, 2, 2+1),
    #         'title': r'good PV'
    #     }

    if var_name is None or var_name == 'goodPV':
        var_dict['goodPV'] = {
            'var': (np.abs(ak.firsts(fb['pv_truth_z']) - ak.firsts(fb['pv_z'])) <= 0.5),
            'bins': np.linspace(0, 2, 2+1),
            'title': r'good PV'
        }
    # # Delta phi (met vs. central jet)
    if var_name is None or var_name == 'dphi_met_central_jet':
        dphi_met_central_jet_tmp = np.arccos(np.cos(fb['met_tst_phi'] - ak.firsts(fb['jet_central_phi'])))
        var_dict['dphi_met_central_jet'] = {
            'var': ak.fill_none(dphi_met_central_jet_tmp, -999),
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(E_T^{miss},\, jet)$'
        }

    # # Counts: constant 0.5 (typically used for normalization)
    # var_dict['counts'] = {
    #     'var': 0.5,
    #     'bins': np.linspace(0, 1, 1+1),
    #     'title': ''
    # }

    # # Jet central timing1
    if var_name is None or var_name == 'jet_central_timing1':
        jet_central_timing1_tmp = ak.firsts(fb['jet_central_timing'])
        var_dict['jet_central_timing1'] = {
            'var': ak.fill_none(jet_central_timing1_tmp, -999),
            'bins': np.linspace(-40, 40, 50+1),
            'title': r'$Jet\ timing$'
        }

    # # Jet central timing (all)
    if var_name is None or var_name == 'jet_central_timing':
        weight_tmp = getWeight(fb, process)
        expanded_weights = ak.flatten(ak.broadcast_arrays(weight_tmp, fb['jet_central_timing'])[0])
        var_dict['jet_central_timing'] = {
            'var': ak.flatten(fb['jet_central_timing']),
            'weight': expanded_weights,
            'bins': np.linspace(-40, 40, 50+1),
            'title': r'$Jet\ timing$'
        }

    # # Jet central EM fraction
    if var_name is None or var_name == 'jet_central_emfrac':
        weight_tmp = getWeight(fb, process)
        expanded_weights = ak.flatten(ak.broadcast_arrays(weight_tmp, fb['jet_central_emfrac'])[0])
        var_dict['jet_central_emfrac'] = {
            'var': ak.flatten(fb['jet_central_emfrac']),
            'bins': np.linspace(-1, 2, 50+1),
            'title': r'$Jet\ EM\ fraction$'
        }

    if var_name is None or var_name == 'jet_central_emfrac':
        jet_central_emfrac1_tmp = ak.firsts(fb['jet_central_emfrac'])
        var_dict['jet_central_emfrac'] = {
            'var': ak.fill_none(jet_central_emfrac1_tmp, -999),
            'bins': np.linspace(-1, 2, 50+1),
            'title': r'$Jet\ EM\ fraction$'
        }


    # Balance: (met_tst_et+ph_pt[0]) divided by the sum over jet_central_pt.
    if var_name is None or var_name == 'balance':
        jet_sum_tmp = ak.sum(fb['jet_central_pt'], axis=-1)
        expr = (fb['met_tst_et'] + ak.firsts(fb['ph_pt'])) / ak.where(jet_sum_tmp != 0, jet_sum_tmp, 1)
        balance = ak.where(jet_sum_tmp != 0, expr, -999) 

        var_dict['balance'] = {
            'var': balance,
            'bins': np.linspace(0, 20, 100+1),
            'title': r'balance'
        }

    if var_name is None or var_name == 'balance_sumet':
        sumet_tmp = fb['met_jetterm_sumet']
        expr = (fb['met_tst_et'] + ak.firsts(fb['ph_pt'])) / ak.where(sumet_tmp != 0, sumet_tmp, 1)
        balance_sumet = ak.where(sumet_tmp != 0, expr, -999)

        var_dict['balance_sumet'] = {
            'var': balance_sumet,
            'bins': np.linspace(0, 80, 80+1),
            'title': r'balance'
        }

    if var_name is None or var_name == 'central_jets_fraction':
        var_dict['central_jets_fraction'] = {
            'var': np.where(fb['n_jet'] > 0, fb['n_jet_central']/fb['n_jet'], -1),
            'bins': np.linspace(-1, 2, 50+1),
            'title': r'Central jets fraction'
        }

    if var_name is None or var_name == 'trigger':
        var_dict['trigger'] = {
            'var': fb['trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M'],
            'bins': np.linspace(0, 2, 2+1),
            'title': r'Pass Trigger'
        }

    # dphi_jj: Use Alt$ logic – if jet_central_phi has at least two entries, compute the difference; else -1.
    # Here we use a Python conditional (this assumes fb['jet_central_phi'] is an array with shape information).
    if var_name is None or var_name == 'dphi_jj':
        phi1_tmp = ak.firsts(fb['jet_central_phi'])
        phi2_tmp = ak.mask(fb['jet_central_phi'], ak.num(fb['jet_central_phi']) >= 2)[:, 1]
        dphi_tmp = np.arccos(np.cos(phi1_tmp - phi2_tmp))
        var_dict['dphi_jj'] = {
            'var': ak.fill_none(dphi_tmp, -999),
            'bins': np.linspace(-1, 4, 20+1),
            'title': r'$\Delta\phi(j1,\, j2)$'
        }
    
    if var_name is None or var_name == 'BDTScore':
        var_dict['BDTScore'] = {
            'var': fb['BDTScore'],
            'bins': np.arange(0, 1+0.1, 0.1),
            'title': 'BDTScore'
        }
    
    return var_dict