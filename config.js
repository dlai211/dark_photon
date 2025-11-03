// Define the cut configurations (same as Python's cut_config)
const cut_config_main = {
    'basic': true, 'metsig': true, 'dphi_met_phterm': true, 'dmet': true,
    'dphi_met_jetterm': true, 'ph_eta': true, 'dphi_jj': true, 'balance': true, 'mt2': true
};
const cut_config_jets_faking_photons3 = {
    'basic': true
}



// Define the variable configurations
const var_config_main = [
    'n_ph',
    'n_ph_baseline',
    'n_el_baseline',
    'n_mu_baseline',
    'n_tau_baseline',
    'mt',
    'metsig',
    'metsigres',
    'met',
    'met_noJVT',
    'dmet',
    'ph_pt',
    'ph_eta',
    'ph_phi',
    'pv_ntracks',
    'jet_central_eta',
    'jet_central_pt1',
    'jet_central_pt2',
    'dphi_met_phterm',
    'dphi_met_jetterm',
    'dphi_phterm_jetterm',
    'failJVT_jet_pt1',
    'softerm',
    'jetterm',
    'jetterm_sumet',
    'n_jet',
    'n_jet_central',
    'n_jet_fwd',
    'central_jets_fraction',
    'balance',
    'dphi_jj',
    'VertexBDTScore',
];
const var_config_jets_faking_photons3 = [
    "balance", "balance_sumet", "VertexBDTScore",
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

// Define the n-1 configureations
const n_1_config_main = [
    "balance", "dmet", "dphi_jj", "dphi_met_jetterm", "dphi_met_phterm", 
    "metsig", "mt", "ph_eta"
];
const n_1_config_jets_faking_photons = [
    "BDTScore", "dphi_met_phterm", "metsig", "ph_eta", "dmet", "dphi_met_jetterm", "balance", "dphi_jj"];



const imageMap_index = {
    "main": {
        cut_config: cut_config_jets_faking_photons3,
        var_config: var_config_main,
        sig_config: sig_config_main,
        n_1_config: n_1_config_main,
        path: [`mc23e/lumi26/`, `mc23e/lumi135/`],
        imagesPerRow: 4
    }, 
    "jets_faking_photons3": {
        cut_config: cut_config_jets_faking_photons3,
        var_config: var_config_jets_faking_photons3,
        sig_config: sig_config_main,
        n_1_config: n_1_config_jets_faking_photons,
        path: [`jets_faking_photons3/lumi26/`, `jets_faking_photons3/lumi135/`],
        imagesPerRow: 4
    },
};