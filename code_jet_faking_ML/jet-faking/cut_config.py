import numpy as np
def getCutDict():
    cut_dict = {}

    cut_dict['dphi_jj'] = {
        'uppercut': np.arange(1, 3.14 + 0.01, 0.01) # dphi_jj < cut
    }
    cut_dict['dphi_phterm_jetterm'] = {
        'lowercut': np.arange(1, 2.5 + 0.05, 0.05), # dphi_phterm_jetterm > cut
        'uppercut': np.arange(2, 4 + 0.1, 0.1) # dphi_phterm_jetterm < cut
    }
    cut_dict['jet_central_eta'] = {
        'lowercut': np.arange(-2.5, 0+0.01, 0.01), # jet_central_eta > cut
        'uppercut': np.arange(0, 2.5+0.01, 0.01) # jet_central_eta < cut
    }
    cut_dict['jet_central_pt2'] = {
        'lowercut': np.arange(20000, 100000+1000, 1000) # jet_central_pt2 > cut
    }
    cut_dict['metsigres'] = {
        'lowercut': np.arange(8600, 15000, 100),
        'uppercut': np.arange(12000, 60000, 100)
    }
    cut_dict['met_noJVT'] = {
        'lowercut': np.arange(50000, 120000, 100),
        'uppercut': np.arange(100000, 250000, 100)
    }
    cut_dict['softerm'] = {
        'uppercut': np.arange(10000, 40000, 100)
    }
    cut_dict['n_jet_central'] = {
        'uppercut': np.arange(0, 8+1, 1) # njet < cut
    }

    return cut_dict
cut_config = getCutDict()
