import numpy as np
from scipy.special import betainc
from scipy.stats import norm

Vars = [
    'metsig', 'metsigres', 'met', 'met_noJVT', 'dmet', 'ph_pt', 'ph_eta', 'ph_phi',
    'jet_central_eta', 'jet_central_pt1', 'jet_central_pt2', 'dphi_met_phterm', 'dphi_met_ph',
    'dphi_met_jetterm', 'dphi_phterm_jetterm', 'dphi_ph_centraljet1', 'metplusph', 'failJVT_jet_pt1',
    'softerm', 'jetterm', 'jetterm_sumet', 'dphi_met_central_jet', 'balance', 'dphi_jj', 'BDTScore', 'n_jet_central'
]

Vars2 = [
    'metsig', 'met', 'met_noJVT', 'dmet', 'dphi_met_phterm','dphi_ph_centraljet1',
    'dphi_phterm_jetterm', 'jetterm', 'dphi_met_central_jet', 'BDTScore', 'weights', 'label', 'process'
]

Vars3 = [
    'metsigres', 'ph_pt', 'ph_eta', 'dphi_met_jetterm', 'failJVT_jet_pt1', 'n_jet_central', 'dphi_jj'
]

Vars_drop = ['weights', 'label', 'process', 'met', 'dphi_phterm_jetterm']

Vars_plot = ['metsig', 'met_noJVT', 'dmet', 'dphi_met_phterm','dphi_ph_centraljet1', 'jetterm', 'dphi_met_central_jet', 
             'BDTScore', 'metsigres', 'ph_pt', 'ph_eta', 'dphi_met_jetterm', 'failJVT_jet_pt1', 'n_jet_central', 'dphi_jj']

variables = [
    "actualIntPerXing", "failJVT_jet_pt", "jet_central_emfrac", "jet_central_eta",
    "jet_central_phi", "jet_central_pt", "jet_central_timing", "met_cst_et",
    "met_jetterm_et", "met_jetterm_phi", "met_jetterm_sumet", "met_phterm_phi",
    "met_softerm_tst_et", "met_tst_et", "met_tst_noJVT_et", "met_tst_phi",
    "met_tst_sig", "met_track_et", "n_ph", "n_ph_baseline", "n_el_baseline",
    "n_mu_baseline", "n_jet", "n_jet_central", "n_tau_baseline", "ph_eta",
    "ph_phi", "ph_pt", "pu_weight", "pv_truth_z", "pv_z",
    "trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M", "vtx_sumPt",
    "mconly_weight", "mc_weight_sum", "xsec_ami", "filter_eff_ami", "kfactor_ami",
    "pu_weight", "jvt_weight", "event"
]
ntuple_name = ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct','gammajet_frag','dijet']
ntuple_name_BDT = ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct','gammajet_frag','dijets']

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

def getVarDict(fb, process, var_name=None):
    var_dict = {}


    if var_name is None or var_name == 'mt':
        var_dict['mt'] = {
            'bins': np.linspace(0, 300, 15+1),
            'title': r'$m_T\ [GeV]$',
        }

    if var_name is None or var_name == 'metsig':
        var_dict['metsig'] = {
            'bins': np.linspace(0, 30, 15+1),
            'title': r'$E_T^{miss}\ significance$',
        }

    if var_name is None or var_name == 'metsigres':
        var_dict['metsigres'] = {
            'bins': np.linspace(0, 100000, 50+1),
            'title': r'$E_T^{miss}\ significance$',
        }

    if var_name is None or var_name == 'met':
        var_dict['met'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ [MeV]$',
        }

    if var_name is None or var_name == 'met_noJVT':
        var_dict['met_noJVT'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}\ [MeV]$'
        }


    if var_name is None or var_name == 'dmet':
        var_dict['dmet'] = {
            'bins': np.linspace(-100000, 100000, 20+1),
            'title': r'$E_{T,\mathrm{noJVT}}^{miss}-E_T^{miss}\ [MeV]$',
        }

    if var_name is None or var_name == 'ph_pt':
        var_dict['ph_pt'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{\gamma}\ [MeV]$',
        }

    if var_name is None or var_name == 'ph_eta':
        var_dict['ph_eta'] = {
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\eta^{\gamma}$'
        }

    if var_name is None or var_name == 'ph_phi':
        var_dict['ph_phi'] = {
            'bins': np.linspace(-4, 4, 50+1),
            'title': r'$\phi^{\gamma}$'
        }

    if var_name is None or var_name == "jet_central_eta":
        var_dict['jet_central_eta'] = {
            'bins': np.linspace(-4, 4, 50+1), 
            'title': r'$\eta^{\mathrm{jets}}$'
        }

    # Jet central pt1 (first jet)
    if var_name is None or var_name == "jet_central_pt1":
        var_dict['jet_central_pt1'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{j1}\ [MeV]$'
        }

    # Jet central pt2 (second jet, if available)
    if var_name is None or var_name == "jet_central_pt2":
        var_dict['jet_central_pt2'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{j2}\ [MeV]$'
        }

    # Jet central pt (all jets)
    if var_name is None or var_name == "jet_central_pt":
        var_dict['jet_central_pt'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{j}\ [MeV]$'
    }

    if var_name is None or var_name == 'dphi_met_phterm':
        var_dict['dphi_met_phterm'] = {
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})$',
        }

    if var_name is None or var_name == 'dphi_met_ph':
        var_dict['dphi_met_ph'] = {
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})$'
        }

    if var_name is None or var_name == 'dphi_met_jetterm':
        var_dict['dphi_met_jetterm'] = {
            'bins': np.linspace(0, 4, 16+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{jet})$'
        }

    if var_name is None or var_name == 'dphi_phterm_jetterm':
        var_dict['dphi_phterm_jetterm'] = {
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(E_T^{\gamma},\, E_T^{jet})$'
        }

    if var_name is None or var_name == 'dphi_met_phterm_minus_dphi_met_jetterm':
        var_dict['dphi_met_phterm_minus_dphi_met_jetterm'] = {
            'bins': np.linspace(-4, 4, 100+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})-\Delta\phi(E_T^{miss},\, E_T^{jet})$'
        }

    if var_name is None or var_name == 'dphi_met_phterm_divide_dphi_met_jetterm':
        var_dict['dphi_met_phterm_divide_dphi_met_jetterm'] = {
            'bins': np.linspace(-4, 4, 100+1),
            'title': r'$\Delta\phi(E_T^{miss},\, E_T^{\gamma})/\Delta\phi(E_T^{miss},\, E_T^{jet})$'
        }


    # Delta phi (photon vs. central jet1)
    if var_name is None or var_name == 'dphi_ph_centraljet1':
        var_dict['dphi_ph_centraljet1'] = {
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(\gamma,\, j1)$'
        }

    # # Delta phi (photon vs. jet1)
    if var_name is None or var_name == 'dphi_ph_jet1':
        var_dict['dphi_ph_jet1'] = {
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(\gamma,\, j1)$'
        }

    # Met plus photon pt
    if var_name is None or var_name == 'metplusph':
        var_dict['metplusph'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{miss}+p_T^{\gamma}\ [MeV]$'
        }

    # # Fail JVT jet pt (all)
    if var_name is None or var_name == 'failJVT_jet_pt':
        var_dict['failJVT_jet_pt'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$p_T^{\mathrm{noJVT\ jet}}\ [MeV]$'
        }

    # # Fail JVT jet pt1 (first element)
    if var_name is None or var_name == 'failJVT_jet_pt1':
        var_dict['failJVT_jet_pt1'] = {
            'bins': np.linspace(20000, 60000, 40+1),
            'title': r'$p_T^{\mathrm{noJVT\ jet1}}\ [MeV]$'
        }

    if var_name is None or var_name == 'softerm':
        var_dict['softerm'] = {
            'bins': np.linspace(0, 100000, 50+1),
            'title': r'$E_T^{soft}\ [MeV]$'
        }

    if var_name is None or var_name == 'jetterm':
        var_dict['jetterm'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{jet}\ [MeV]$'
        }

    if var_name is None or var_name == 'jetterm_sumet':
        var_dict['jetterm_sumet'] = {
            'bins': np.linspace(0, 300000, 50+1),
            'title': r'$E_T^{jet}\ [MeV]$'
        }

    if var_name is None or var_name == 'n_jet_central':
        var_dict['n_jet_central'] = {
            'bins': np.linspace(0, 10, 10+1),
            'title': r'$N_{jet}^{central}$'
        }

    # # Delta phi (met vs. central jet)
    if var_name is None or var_name == 'dphi_met_central_jet':
        var_dict['dphi_met_central_jet'] = {
            'bins': np.linspace(0, 4, 50+1),
            'title': r'$\Delta\phi(E_T^{miss},\, jet)$'
        }

    # # Jet central timing1
    if var_name is None or var_name == 'jet_central_timing1':
        var_dict['jet_central_timing1'] = {
            'bins': np.linspace(-40, 40, 50+1),
            'title': r'$Jet\ timing$'
        }

    # # Jet central EM fraction
    if var_name is None or var_name == 'jet_central_emfrac':
        var_dict['jet_central_emfrac'] = {
            'bins': np.linspace(-1, 2, 50+1),
            'title': r'$Jet\ EM\ fraction$'
        }

    if var_name is None or var_name == 'jet_central_emfrac':
        var_dict['jet_central_emfrac'] = {
            'bins': np.linspace(-1, 2, 50+1),
            'title': r'$Jet\ EM\ fraction$'
        }


    # Balance: (met_tst_et+ph_pt[0]) divided by the sum over jet_central_pt.
    if var_name is None or var_name == 'balance':
        var_dict['balance'] = {
            'bins': np.linspace(0, 20, 100+1),
            'title': r'balance'
        }

    if var_name is None or var_name == 'balance_sumet':
        var_dict['balance_sumet'] = {
            'bins': np.linspace(0, 80, 80+1),
            'title': r'balance'
        }

    if var_name is None or var_name == 'central_jets_fraction':
        var_dict['central_jets_fraction'] = {
            'bins': np.linspace(-1, 2, 50+1),
            'title': r'Central jets fraction'
        }


    # dphi_jj: Use Alt$ logic – if jet_central_phi has at least two entries, compute the difference; else -1.
    # Here we use a Python conditional (this assumes fb['jet_central_phi'] is an array with shape information).
    if var_name is None or var_name == 'dphi_jj':
        var_dict['dphi_jj'] = {
            'bins': np.linspace(-1, 4, 20+1),
            'title': r'$\Delta\phi(j1,\, j2)$'
        }
    
    if var_name is None or var_name == 'BDTScore':
        var_dict['BDTScore'] = {
            'bins': np.arange(0, 1+0.1, 0.1),
            'title': 'BDTScore'
        }

    if var_name is None or var_name == 'bdtscore2':
        var_dict['bdtscore2'] = {
            'bins': np.arange(0, 0.3+0.01, 0.01),
            'title': 'BDTScore'
        }
    
    return var_dict