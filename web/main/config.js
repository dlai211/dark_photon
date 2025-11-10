// Define the cut configurations (same as Python's cut_config)
const cut_config_main = {
    'basic': true, 'metsig': true, 'dphi_met_phterm': true, 'dmet': true,
    'dphi_met_jetterm': true, 'ph_eta': true, 'dphi_jj': true, 'balance': true, 'mt2': true
};
const cut_config_jets_faking_photons3 = {
    'basic': true, 'selection': true
}



// Define the variable configurations
const var_config_main = [
    ['mt', 'balance', 'VertexBDTScore'],
    ['n_ph', 'n_ph_baseline', 'n_el', 'n_el_baseline', 'n_mu_baseline', 'n_tau_baseline', 'pv_ntracks', 'n_jet', 
     'n_jet_central', 'n_jet_fwd', 'central_jets_fraction'],
    ['ph_pt', 'ph_eta', 'ph_phi'],
    ['jet_central_eta', 'jet_central_pt1', 'jet_central_pt2', 'jet_central_vecSumPt', 'failJVT_jet_pt1', 'failJVT_jet_vecSumPt'],
    ['met', 'met_noJVT', 'metsig', 'metsigres', 'dmet', 'softerm', 'jetterm', 'jetterm_sumet'], 
    ['dphi_met_phterm', 'dphi_met_jetterm', 'dphi_phterm_jetterm', 'dphi_jj']
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


// Define group
const group_titles_main = [
  "Composite / Analysis Variables",
  "Number of []",
  "Photon / Electron Kinematics",
  "Jet Kinematics",
  "Missing ET",
  "Angular Separations"
];


const imageMap_index = {
    "main": {
        cut_config: cut_config_jets_faking_photons3,
        var_config: var_config_main,
        sig_config: sig_config_main,
        n_1_config: n_1_config_main,
        path: [`main/mt80/`, `main/mt100_140/`],
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

imageMap_index.main.group_titles = group_titles_main;
