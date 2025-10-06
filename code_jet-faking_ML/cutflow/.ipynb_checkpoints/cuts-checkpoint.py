# ---------- INTERNAL SELECTION CUT ------------

    metsig_tmp = fb['met_tst_sig'] 
    mask1 = metsig_tmp > 6
    fb = fb[mask1]
    log_step(ntuple_name, step, "met_tst_sig>6", fb, start_time); step += 1
    
    ph_eta_tmp = np.abs(ak.firsts(fb['ph_eta']))
    fb = fb[ph_eta_tmp < 1.75]
    log_step(ntuple_name, step, "ph_eta<1.75", fb, start_time); step += 1

    dphi_met_phterm_tmp = np.arccos(np.cos(fb['met_tst_phi'] - fb['met_phterm_phi'])) # added cut 3
    fb = fb[dphi_met_phterm_tmp > 1.25]
    log_step(ntuple_name, step, "dphi_met_phterm>1.25", fb, start_time); step += 1

    dmet_tmp = fb['met_tst_noJVT_et'] - fb['met_tst_et']
    mask1 = dmet_tmp > -10000
    fb = fb[mask1]
    log_step(ntuple_name, step, "dmet>-10GeV", fb, start_time); step += 1

    dphi_jj_tmp = fb['dphi_central_jj']
    dphi_jj_tmp = ak.where(dphi_jj_tmp == -10, np.nan, dphi_jj_tmp)
    dphi_jj_tmp = np.arccos(np.cos(dphi_jj_tmp))
    dphi_jj_tmp = ak.where(np.isnan(dphi_jj_tmp), -999, dphi_jj_tmp)
    fb = fb[dphi_jj_tmp < 2.5]
    log_step(ntuple_name, step, "dphi_jj_central<2.5", fb, start_time); step += 1

    dphi_met_jetterm_tmp = np.where(fb['met_jetterm_et'] != 0,   # added cut 5
                        np.arccos(np.cos(fb['met_tst_phi'] - fb['met_jetterm_phi'])),
                        -999)
    fb = fb[dphi_met_jetterm_tmp < 0.75]
    log_step(ntuple_name, step, "dphi_met_jetterm<0.75", fb, start_time); step += 1

# ---------- SELECTION CUT ------------

    metsig_tmp = fb['met_tst_sig'] 
    mask1 = metsig_tmp > 7
    # fb = fb[mask1]
    # log_step(ntuple_name, step, "met_tst_sig>7", fb, start_time); step += 1
    mask2 = metsig_tmp < 20
    fb = fb[mask1 * mask2]
    log_step(ntuple_name, step, "20>met_tst_sig>7", fb, start_time); step += 1
    
    ph_eta_tmp = np.abs(ak.firsts(fb['ph_eta']))
    fb = fb[ph_eta_tmp < 1.74]
    log_step(ntuple_name, step, "ph_eta<1.74", fb, start_time); step += 1

    dphi_met_phterm_tmp = np.arccos(np.cos(fb['met_tst_phi'] - fb['met_phterm_phi'])) # added cut 3
    fb = fb[dphi_met_phterm_tmp > 1.23]
    log_step(ntuple_name, step, "dphi_met_phterm>1.23", fb, start_time); step += 1

    dmet_tmp = fb['met_tst_noJVT_et'] - fb['met_tst_et']
    mask1 = dmet_tmp > -21200
    # fb = fb[mask1]
    # log_step(ntuple_name, step, "dmet>-21GeV", fb, start_time); step += 1
    mask2 = dmet_tmp < 41600
    fb = fb[mask1 * mask2]
    log_step(ntuple_name, step, "41.9GeV>dmet>-21.2GeV", fb, start_time); step += 1

    dphi_jj_tmp = fb['dphi_central_jj']
    dphi_jj_tmp = ak.where(dphi_jj_tmp == -10, np.nan, dphi_jj_tmp)
    dphi_jj_tmp = np.arccos(np.cos(dphi_jj_tmp))
    dphi_jj_tmp = ak.where(np.isnan(dphi_jj_tmp), -999, dphi_jj_tmp)
    fb = fb[dphi_jj_tmp < 2.37]
    log_step(ntuple_name, step, "dphi_jj_central<2.37", fb, start_time); step += 1

    dphi_met_jetterm_tmp = np.where(fb['met_jetterm_et'] != 0,   # added cut 5
                        np.arccos(np.cos(fb['met_tst_phi'] - fb['met_jetterm_phi'])),
                        -999)
    fb = fb[dphi_met_jetterm_tmp < 0.73]
    log_step(ntuple_name, step, "dphi_met_jetterm<0.73", fb, start_time); step += 1

    # dphi_met_central_jet_tmp = np.arccos(np.cos(fb['met_tst_phi'] - ak.firsts(fb['jet_central_phi'])))
    # dphi_met_central_jet_tmp = ak.fill_none(dphi_met_central_jet_tmp, -999)
    # mask1 = dphi_met_central_jet_tmp == -999
    # mask2 = dphi_met_central_jet_tmp > 2.23
    # mask = mask1 | mask2
    # fb = fb[mask]
    # log_step(ntuple_name, step, "dphi_met_central_jet>2.23", fb, start_time); step += 1

    # ---------- FURTHER SELECTION CUT ------------
    jet_sum_tmp = ak.sum(fb['jet_central_pt'], axis=-1)
    expr = (fb['met_tst_et'] + ak.firsts(fb['ph_pt'])) / ak.where(jet_sum_tmp != 0, jet_sum_tmp, 1)
    balance_tmp = ak.where(jet_sum_tmp != 0, expr, -999)
    mask_nan = balance_tmp == -999
    mask = mask_nan | (balance_tmp > 0.91)
    fb = fb[mask]
    log_step(ntuple_name, step, "balance>0.91", fb, start_time); step += 1

    fb = fb[fb['met_jetterm_et'] > 92000]
    log_step(ntuple_name, step, "jetterm>92GeV", fb, start_time); step += 1

    dphi_phterm_jetterm_tmp = np.where(fb['met_jetterm_et'] > 0,
                                        np.arccos(np.cos(fb['met_phterm_phi'] - fb['met_jetterm_phi'])),
                                        -999)
    mask1 = dphi_phterm_jetterm_tmp == -999
    mask2 = dphi_phterm_jetterm_tmp > 1.6 
    mask3 = dphi_phterm_jetterm_tmp < 3.1 
    in_window = mask2 & mask3
    mask = mask1 | in_window
    fb = fb[mask]
    log_step(ntuple_name, step, "3.1>dphi_phterm_jetterm>1.6", fb, start_time); step += 1

    metsigres_tmp = fb['met_tst_et'] / fb['met_tst_sig']
    fb = fb[metsigres_tmp < 36000]
    log_step(ntuple_name, step, "metsigres<36000", fb, start_time); step += 1

    fb = fb[fb['met_tst_noJVT_et'] > 90000]
    log_step(ntuple_name, step, "met_noJVT>90GeV", fb, start_time); step += 1



















# ------------------ SHARED CUTS -----------------------

# Internal Notes
shared_cuts = [
    'n_mu_baseline==0',
    'n_el_baseline==0',
    'n_tau_baseline==0',
    'trigger==1',
    'ph_pt>50GeV',
    'MET>100GeV',
    'n_jet_central<=3',
    '140>mT>100GeV',
    'VertexBDTScore>0.1',
    'met_tst_sig>6',
    'ph_eta<1.75',
    'dphi_met_phterm>1.25',
    'dmet>-10GeV',
    'dphi_jj_central<2.5',
    'dphi_met_jetterm<0.75'
]

# selection_1 (80)
shared_cuts = [
    'n_mu_baseline==0',
    'n_el_baseline==0',
    'n_tau_baseline==0',
    'trigger==1',
    'ph_pt>50GeV',
    'MET>100GeV',
    'n_jet_central<=3',
    'mT>80GeV',
    'VertexBDTScore>0.1',
    'met_tst_sig>7',
    'ph_eta<1.75',
    'dphi_met_phterm>1.35',
    'dmet>-20GeV',
    'dphi_jj_central<2.5',
    'dphi_met_jetterm<0.75'
]

# detail_2 (80)
shared_cuts = [
    'n_mu_baseline==0',
    'n_el_baseline==0',
    'n_tau_baseline==0',
    'trigger==1',
    'ph_pt>50GeV',
    'MET>100GeV',
    'n_jet_central<=3',
    'mT>80GeV',
    'VertexBDTScore>0.1',
    'met_tst_sig>7',
    'ph_eta<1.75',
    'dphi_met_phterm>1.35',
    'dmet>-20GeV',
    'dphi_jj_central<2.5',
    'dphi_met_jetterm<0.75',
    'balance>0.90',
    'jetterm>80GeV',
    '3.1>dphi_phterm_jetterm>1.8',
    'met_noJVT>90GeV'
]

# selection_1 (100, 140)
shared_cuts = [
    'n_mu_baseline==0',
    'n_el_baseline==0',
    'n_tau_baseline==0',
    'trigger==1',
    'ph_pt>50GeV',
    'MET>100GeV',
    'n_jet_central<=3',
    '140>mT>100GeV',
    'VertexBDTScore>0.1',
    'met_tst_sig>7',
    'ph_eta<1.75',
    'dphi_met_phterm>1.20',
    'dmet>-20GeV',
    'dphi_jj_central<2.35',
    'dphi_met_jetterm<0.85'
]

# detail_2 (100, 140)
shared_cuts = [
    'n_mu_baseline==0',
    'n_el_baseline==0',
    'n_tau_baseline==0',
    'trigger==1',
    'ph_pt>50GeV',
    'MET>100GeV',
    'n_jet_central<=3',
    '140>mT>100GeV',
    'VertexBDTScore>0.1',
    'met_tst_sig>7',
    'ph_eta<1.75',
    'dphi_met_phterm>1.20',
    'dmet>-20GeV',
    'dphi_jj_central<2.35',
    'dphi_met_jetterm<0.85',
    'balance>0.90',
    'jetterm>80GeV',
    '3.1>dphi_phterm_jetterm>1.6',
    'metsigres<42000',
    'met_noJVT>90GeV'
]