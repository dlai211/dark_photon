// Define the cut configurations (same as Python's cut_config)
const cut_config = {
    'basic': true, 'metsig': true, 'dphi_met_phterm': true, 'dmet': true,
    'dphi_met_jetterm': true, 'ph_eta': true, 'dphi_jj': true, 'balance': true, 'mt2': true
};

const var_config = [
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

const sig_config = [
    "BDTScore", "balance", "dmet", "dphi_jj", "dphi_met_jetterm", 
    "dphi_met_phterm", "dphi_ph_centraljet1", "dphi_phterm_jetterm", 
    "met", "metsig", "mt", "n_jet_central", "ph_eta", "ph_pt"
];

const n_1_config = [
    "balance", "dmet", "dphi_jj", "dphi_met_jetterm", "dphi_met_phterm", 
    "metsig", "mt", "ph_eta"
];

// Function to generate image paths dynamically
function generateImagePaths(cut_name, mode, lumi) {
    if (!cut_config[cut_name] && mode !== "n-1") return [];
    
    let images = [];
    let path = (lumi === "26fb") ? `main/lumi26/` : (lumi === "135fb") ? `main/lumi135/` : ``;

    if (mode == "performance") {
        var_config.forEach((var_name) => {
            images.push(path + `mc23d_${cut_name}cut/${var_name}_nodijet.png`);
            images.push(path + `mc23d_${cut_name}cut/roc_curve_${var_name}.png`);
        })
    } else if (mode == "significance") {
        sig_config.forEach((sig_name) => {
            images.push(path + `mc23d_${cut_name}cut/${sig_name}_nodijet.png`);
            images.push(path + `mc23d_${cut_name}cut/significance_${sig_name}_lowercut.png`);
            images.push(path + `mc23d_${cut_name}cut/significance_${sig_name}_uppercut.png`);
        })
    } else if (mode == "n-1") {
        n_1_config.forEach((n_1_name) => {
            images.push(path + `mc23d_n-1cut/${n_1_name}_nodijet.png`);
            images.push(path + `mc23d_n-1cut/significance_${n_1_name}_lowercut.png`);
            images.push(path + `mc23d_n-1cut/significance_${n_1_name}_uppercut.png`);
        })
    }

    return images;
}