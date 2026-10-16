import ROOT
import time

# --- 1. Weights ---
ROOT.gInterpreter.Declare("""
float get_sf_jet(float eta, bool is2023) {
    float aeta = std::abs(eta);
    if (is2023) {
        if (aeta <= 0.6) return 2.102;
        if (aeta <= 1.37) return 1.875;
        if (aeta <= 1.52) return 0.0;
        if (aeta <= 1.81) return 1.911;
        if (aeta <= 2.37) return 2.382;
    } else {
        if (aeta <= 0.6) return 1.453;
        if (aeta <= 1.37) return 1.957;
        if (aeta <= 1.52) return 0.0;
        if (aeta <= 1.81) return 1.894;
        if (aeta <= 2.37) return 1.605;
    }
    return 0.0;
}

float get_sf_el(float pt, float eta, bool is2023) {
    float aeta = std::abs(eta);
    float scale = 0.0; float norm = 0.0;
    if (is2023) {
        if (pt > 50000 && pt <= 52000) scale = 1.257;
        else if (pt <= 54000) scale = 1.040; else if (pt <= 56000) scale = 1.104;
        else if (pt <= 58000) scale = 1.064; else if (pt <= 60000) scale = 1.079;
        else if (pt <= 62000) scale = 1.113; else if (pt <= 65000) scale = 1.088;
        else if (pt <= 70000) scale = 1.077; else if (pt <= 80000) scale = 1.074;
        else if (pt <= 100000) scale = 1.099; else if (pt <= 200000) scale = 1.097;
        if (aeta <= 0.60) norm = 0.02695; else if (aeta <= 1.37) norm = 0.02822;
        else if (aeta <= 1.52) norm = 0.00000; else if (aeta <= 1.81) norm = 0.05412;
        else if (aeta <= 2.37) norm = 0.10188;
    } else {
        if (pt > 50000 && pt <= 52000) scale = 1.318;
        else if (pt <= 54000) scale = 1.103; else if (pt <= 56000) scale = 1.118;
        else if (pt <= 58000) scale = 1.134; else if (pt <= 60000) scale = 1.107;
        else if (pt <= 62000) scale = 1.129; else if (pt <= 65000) scale = 1.137;
        else if (pt <= 70000) scale = 1.105; else if (pt <= 80000) scale = 1.125;
        else if (pt <= 100000) scale = 1.123; else if (pt <= 200000) scale = 1.109;
        if (aeta <= 0.60) norm = 0.03734; else if (aeta <= 1.37) norm = 0.02868;
        else if (aeta <= 1.52) norm = 0.00000; else if (aeta <= 1.81) norm = 0.04860;
        else if (aeta <= 2.37) norm = 0.09252;
    }
    return scale * norm;
}
""")

# --- 2. Cuts ---
mc_cuts = [
    ("Initial", "true"),
    ("n_bjet == 0", "n_bjet == 0"),
    ("n_ph == 1", "n_ph == 1"),
    ("Photon Iso", "(ph_topoetcone40[0]-2450.)/ph_pt[0] < 0.022"),
    ("Lepton Veto", "n_mu_baseline == 0 && n_el_baseline == 0 && n_tau_baseline == 0"),
    ("Trigger", "trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M == 1"),
    ("MET > 100", "met_tst_et * 0.001 > 100"),
    ("ph_pt > 50", "ph_pt[0] * 0.001 > 50"),
    ("n_jet <= 3", "n_jet_central <= 3"),
    ("Vertex BDT > 0.1", "BDTScore > 0.1"),
    ("MT > 80", "sqrt(2.0*ph_pt[0]*met_tst_et*(1.0 - cos(ph_phi[0] - met_tst_phi)))*0.001 > 80"),
    ("ML BDT > 0.02", "MLBDTScore > 0.02")
]

jfy_cuts = [
    ("Initial", "true"),
    ("n_bjet == 0", "n_bjet == 0"),
    ("n_ph_bsl == 1", "n_ph_baseline == 1"),
    ("Photon Anti-Iso", "(ph_topoetcone40[0]-2450.)/ph_pt[0] > 0.1"),
    ("Lepton Veto", "n_mu_baseline == 0 && n_el_baseline == 0 && n_tau_baseline == 0"),
    ("Trigger", "trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M == 1"),
    ("MET > 100", "met_tst_et * 0.001 > 100"),
    ("ph_pt > 50", "ph_pt[0] * 0.001 > 50"),
    ("n_jet <= 3", "n_jet_central <= 3"),
    ("Vertex BDT > 0.1", "BDTScore > 0.1"),
    ("MT > 80", "sqrt(2.0*ph_pt[0]*met_tst_et*(1.0 - cos(ph_phi[0] - met_tst_phi)))*0.001 > 80"),
    ("ML BDT > 0.02", "MLBDTScore > 0.02")
]

efy_cuts = [
    ("Initial", "true"),
    ("n_el == 1", "n_el == 1"),
    ("n_ph_bsl == 0", "n_ph_baseline == 0"),
    ("n_bjet == 0", "n_bjet == 0"),
    ("Trig Single El", "trigger_single_el == 1"),
    ("Trig MET+G", "trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M == 1"),
    ("MET > 100", "met_tst_et * 0.001 > 100"),
    ("MT_el > 80", "sqrt(2.0*el_pt[0]*met_tst_et*(1.0 - cos(el_phi[0] - met_tst_phi)))*0.001 > 80"),
    ("Vertex BDT > 0.1", "BDTScore > 0.1"),
    ("ML BDT >= 0.02", "MLBDTScore >= 0.02")
]

# --- 3. Run Analysis ---
def run_combined_analysis():
    ROOT.EnableImplicitMT()
    
    samples_mc = {
        "ggHyyd": ["mc23d_ggHyyd_y", "mc23e_ggHyyd_y"],
        "VBFHyyd": ["mc23d_VBFHyyd_y", "mc23e_VBFHyyd_y"],
        "WHyyd": ["mc23d_WHyyd_y", "mc23e_WHyyd_y"],
        "ZHyyd": ["mc23d_ZHyyd_y", "mc23e_ZHyyd_y"],
        "Wgamma": ["mc23d_Wgamma_y", "mc23e_Wgamma_y"],
        "Zgamma": ["mc23d_Zgamma_y", "mc23e_Zgamma_y"],
        "gammajet_dir": ["mc23d_gammajet_direct_y", "mc23e_gammajet_direct_y"]
    }
    
    samples_efy = {"e->gamma": ["data23_eprobe", "data24_eprobe"]}
    samples_jfy = {"j->gamma": ["data23_y", "data24_y"]}

    results_mc = {n: {c[0]: 0.0 for c in mc_cuts} for n in samples_mc.keys()}
    results_efy = {n: {c[0]: 0.0 for c in efy_cuts} for n in samples_efy.keys()}
    results_jfy = {n: {c[0]: 0.0 for c in jfy_cuts} for n in samples_jfy.keys()}

    # --- MC Loop ---
    for process, files in samples_mc.items():
        for f_name in files:
            lumi = 26000.0 if "mc23d" in f_name else 109000.0
            df = ROOT.RDataFrame("nominal", f"/data/jlai/df_ntups_w_bdt_new/{f_name}_nominal_bdt_withMLScores.root")
            is_sig = any(s in f_name for s in ["ggHyyd", "WH", "VBF", "ZH"])
            xsec = "0.052" if is_sig else "xsec_ami"
            formula = f"(mconly_weight / mc_weight_sum) * {xsec} * pu_weight * jvt_weight * filter_eff_ami * kfactor_ami * 1000.0 * {lumi} {'* 0.01' if is_sig else ''}"
            df = df.Define("weight", formula)
            tmp_df = df
            for label, cut_str in mc_cuts:
                tmp_df = tmp_df.Filter(cut_str)
                results_mc[process][label] += tmp_df.Sum("weight").GetValue()

    # --- EFY Loop ---
    for process, files in samples_efy.items():
        for f_name in files:
            is23 = "23" in f_name
            df = ROOT.RDataFrame("nominal", f"/data/jlai/df_ntups_w_bdt_new/{f_name}_nominal_bdt_withMLScores.root")
            df = df.Define("weight", f"get_sf_el(el_pt[0], el_eta[0], {'true' if is23 else 'false'})")
            tmp_df = df
            for label, cut_str in efy_cuts:
                tmp_df = tmp_df.Filter(cut_str)
                results_efy[process][label] += tmp_df.Sum("weight").GetValue()

    # --- JFY Loop ---
    for process, files in samples_jfy.items():
        for f_name in files:
            is23 = "23" in f_name
            df = ROOT.RDataFrame("nominal", f"/data/jlai/df_ntups_w_bdt_new/{f_name}_nominal_bdt_withMLScores.root")
            df = df.Define("weight", f"get_sf_jet(ph_eta[0], {'true' if is23 else 'false'})")
            tmp_df = df
            for label, cut_str in jfy_cuts:
                tmp_df = tmp_df.Filter(cut_str)
                results_jfy[process][label] += tmp_df.Sum("weight").GetValue()

    # --- Printing ---
    print("\n--- SIGNAL AND MC BACKGROUNDS ---")
    header = f"{'Cut Step':<20}" + "".join([f" | {p:>12}" for p in samples_mc.keys()])
    print(header + "\n" + "-"*len(header))
    for l, _ in mc_cuts:
        print(f"{l:<20}" + "".join([f" | {results_mc[p][l]:>12.2f}" for p in samples_mc.keys()]))

    print("\n--- ELECTRON FAKING PHOTON (EFY) ---")
    print(f"{'Cut Step':<20} | {'e->gamma':>12}\n" + "-"*35)
    for l, _ in efy_cuts:
        print(f"{l:<20} | {results_efy['e->gamma'][l]:>12.2f}")

    print("\n--- JET FAKING PHOTON (JFY) ---")
    print(f"{'Cut Step':<20} | {'j->gamma':>12}\n" + "-"*35)
    for l, _ in jfy_cuts:
        print(f"{l:<20} | {results_jfy['j->gamma'][l]:>12.2f}")

if __name__ == "__main__":
    run_combined_analysis()