import numpy as np
signal_name = 'ggHyyd'  # Define signal dataset
cut_name = 'basic'

def getCutDict():
    cut_dict = {}
    # Reduced Features
    cut_dict['balance'] = {
        'lowercut': np.arange(0, 1.5 + 0.01, 0.01), # balance > cut
        'uppercut': np.arange(1.5, 9, 0.05) # balance < cut
    }
    cut_dict['dmet'] = {
        'lowercut': np.arange(-30000, 10000 + 100, 100), # dmet > cut
        'uppercut': np.arange(10000, 100000 + 100, 100), # -10000 < dmet < cut
    }
    cut_dict['metsig'] = {
        'lowercut': np.arange(0, 10 + 1, 1), # metsig > cut
        'uppercut': np.arange(10, 30 + 1, 1), # metsig < cut 
    }
    cut_dict['jetterm'] = {
        'lowercut': np.arange(0, 150000+500, 500) # jetterm > cut
    }
    cut_dict['dphi_met_phterm'] = {
        'lowercut': np.arange(1, 2 + 0.01, 0.01), # dphi_met_phterm > cut
    }
    cut_dict['dphi_met_central_jet'] = {
        'lowercut': np.arange(1.5, 2.8, 0.01)
    }
    cut_dict['ph_eta'] = {
        'uppercut': np.arange(1, 2.5 + 0.01, 0.01), # ph_eta < cut
    }
    cut_dict['ph_pt'] = {
        'lowercut': np.arange(50000, 100000 + 1000, 1000),  # ph_pt > cut
        'uppercut': np.arange(100000, 300000 + 1000, 1000),  # ph_pt > cut
    }

    # Other Features
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

    return cut_dict
cut_config = getCutDict()