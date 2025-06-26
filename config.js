// Define the cut configurations (same as Python's cut_config)
const cut_config_main = {
    'basic': true, 'metsig': true, 'dphi_met_phterm': true, 'dmet': true,
    'dphi_met_jetterm': true, 'ph_eta': true, 'dphi_jj': true, 'balance': true, 'mt2': true
};
const cut_config_mt100 = {
    'basic': true, 'metsig': true, 'dphi_met_phterm': true, 'dmet': true,
    'dphi_met_jetterm': true, 'ph_eta': true, 'dphi_jj': true, 'balance': true
};
const cut_config_dphi_diff = {
    'basic': true, 'dphi_met_phterm_minus_dphi_met_jetterm': true, 'balance': true, 'metsig': true, 
    'ph_eta': true, 'dmet': true,  'dphi_jj': true, 'metsig2': true, 'dphi_met_jetterm': true
};
const cut_config_sig = {
    'basic': true, 'selection': true
}
const cut_config_jets_faking_photons = {
    'basic': true, 'BDTScore': true, 'dphi_met_phterm': true, 'metsig': true, 'ph_eta': true,
    'dmet': true, 'dphi_met_jetterm': true, "balance": true, 'dphi_jj': true
}



// Define the variable configurations
const var_config_main = [
    "actualIntPerXing", "balance", "balance_sumet", "BDTScore",
    "central_jets_fraction", "dmet", "dphi_jj", "dphi_met_central_jet", 
    "dphi_met_jetterm", "dphi_met_ph", "dphi_met_phterm", "dphi_ph_jet1", 
    "dphi_ph_centraljet1", "dphi_phterm_jetterm", "failJVT_jet_pt", 
    "failJVT_jet_pt1", "goodPV", "jet_central_emfrac", "jet_central_eta", 
    "jet_central_pt", "jet_central_pt1", "jet_central_pt2", "jet_central_timing", 
    "jet_central_timing1", "jetterm", "jetterm_sumet", "met", "met_cst", 
    "met_noJVT", "met_track", "metplusph", "metsig", "metsigres", 
    "mt", "n_jet", "n_jet_central", "n_jet_fwd", "n_el_baseline", 
    "n_mu_baseline", "n_ph", "n_ph_baseline", "n_tau_baseline", "ph_eta", 
    "ph_phi", "ph_pt", "puWeight", "softerm", "trigger", "vtx_sumPt"
];
const var_config_jets_faking_photons = [
    "balance", "balance_sumet", "BDTScore",
    "central_jets_fraction", "dmet", "dphi_jj", "dphi_met_central_jet", 
    "dphi_met_jetterm", "dphi_met_ph", "dphi_met_phterm", "dphi_ph_jet1", 
    "dphi_ph_centraljet1", "dphi_phterm_jetterm", "failJVT_jet_pt", 
    "failJVT_jet_pt1", "jet_central_emfrac", "jet_central_eta", 
    "jet_central_pt", "jet_central_pt1", "jet_central_pt2", "jet_central_timing", 
    "jet_central_timing1", "jetterm", "jetterm_sumet", "met", "met_cst", 
    "met_noJVT", "met_track", "metplusph", "metsig", "metsigres", 
    "mt", "n_jet", "n_jet_central", "n_jet_fwd", "n_el_baseline", 
    "n_mu_baseline", "n_ph", "n_ph_baseline", "n_tau_baseline", "ph_eta", 
    "ph_phi", "ph_pt", "softerm", "trigger", "vtx_sumPt"
];

// Define the significance configurations
const sig_config_main = [
    "BDTScore", "balance", "dmet", "dphi_jj", "dphi_met_jetterm", 
    "dphi_met_phterm", "dphi_ph_centraljet1", "dphi_phterm_jetterm", 
    "met", "metsig", "mt", "n_jet_central", "ph_eta", "ph_pt"
];
const sig_config_dphi_diff = [
    "BDTScore", "balance", "dmet", "dphi_jj", "dphi_met_phterm_minus_dphi_met_jetterm", 
    "dphi_met_jetterm", "dphi_met_phterm", "dphi_ph_centraljet1", "dphi_phterm_jetterm", 
    "met", "metsig", "mt", "n_jet_central", "ph_eta", "ph_pt"
];


// Define the n-1 configureations
const n_1_config_main = [
    "balance", "dmet", "dphi_jj", "dphi_met_jetterm", "dphi_met_phterm", 
    "metsig", "mt", "ph_eta"
];
const n_1_config_mt100 = [
    "balance", "dmet", "dphi_jj", "dphi_met_jetterm", "dphi_met_phterm", 
    "metsig", "ph_eta"
];
const n_1_config_dphi_diff = ["dphi_met_phterm_minus_dphi_met_jetterm", 
    "balance", "metsig", "ph_eta", "dmet",  "dphi_jj", "dphi_met_jetterm"];
const n_1_config_jets_faking_photons = [
    "BDTScore", "dphi_met_phterm", "metsig", "ph_eta", "dmet", "dphi_met_jetterm", "balance", "dphi_jj"];



const imageMap_index = {
    "main": {
        cut_config: cut_config_main,
        var_config: var_config_main,
        sig_config: sig_config_main,
        n_1_config: n_1_config_main,
        path: [`main/lumi26/`, `main/lumi135/`],
        imagesPerRow: 4
    }, 
    "mt100": {
        cut_config: cut_config_mt100,
        var_config: var_config_main,
        sig_config: sig_config_main,
        n_1_config: n_1_config_mt100,
        path: [``, `test/lumi135/`],
        imagesPerRow: 4
    }, 
    "dphi_diff": {
        cut_config: cut_config_dphi_diff,
        var_config: var_config_main,
        sig_config: sig_config_dphi_diff,
        n_1_config: n_1_config_dphi_diff,
        path: [``, `test/dphi_diff/`],
        imagesPerRow: 4
    },
    "significance": {
        cut_config: cut_config_sig,
        var_config: var_config_main,
        sig_config: sig_config_main,
        n_1_config: n_1_config_mt100,
        path: [``, `test/sigstudy/`],
        imagesPerRow: 4
    },
    "jets_faking_photons": {
        cut_config: cut_config_jets_faking_photons,
        var_config: var_config_jets_faking_photons,
        sig_config: sig_config_main,
        n_1_config: n_1_config_jets_faking_photons,
        path: [`jets_faking_photons/lumi26/`, `jets_faking_photons/lumi135/`],
        imagesPerRow: 4
    }, 
    "electrons_faking_photons": {
        cut_config: cut_config_jets_faking_photons,
        var_config: var_config_jets_faking_photons,
        sig_config: sig_config_main,
        n_1_config: n_1_config_main,
        path: [`electrons_faking_photons/lumi26/`, `electrons_faking_photons/lumi135/`],
        imagesPerRow: 4
    }, 
};