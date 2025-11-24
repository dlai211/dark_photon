#!/usr/bin/env python3
import math
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# ------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------

base_path = "/data/fpiazza/ggHyyd/NtuplesWithBDTSkim"

# The samples in the order you used in the matplotlib code
samples = [
    "data",            # SR data (tight isolation)
    "Zgamma",          # Z(νν)+γ
    "Wgamma",          # W(ℓν)+γ
    "data_y",          # jet -> γ
    "data_eprobe",     # e -> γ
    "gammajet_direct", # γ+jets direct
]

# Map to file, TTree name and "type" which controls the selection
sample_cfg = {
    "data": {
        "file": "data24_y",
        "tree": "nominal",
        "type": "data_sr",
    },
    "Zgamma": {
        "file": "mc23e_Zgamma_y",
        "tree": "nominal",
        "type": "mc",
    },
    "Wgamma": {
        "file": "mc23e_Wgamma_y",
        "tree": "nominal",
        "type": "mc",
    },
    "gammajet_direct": {
        "file": "mc23e_gammajet_direct_y",
        "tree": "gammajets",
        "type": "mc",
    },
    "data_y": {
        "file": "data24_y",
        "tree": "nominal",
        "type": "jetfake",
    },
    "data_eprobe": {
        "file": "data24_eprobe",
        "tree": "nominal",
        "type": "eprobe",
    },
}

# Colors & legend labels as in your plot_config.py
colors = {
    "data": ROOT.kBlack,
    "Zgamma": ROOT.TColor.GetColor("#ff6600"),
    "Wgamma": ROOT.TColor.GetColor("#ff9933"),
    "data_eprobe": ROOT.TColor.GetColor("#339933"),
    "data_y": ROOT.TColor.GetColor("#00cccc"),
    "gammajet_direct": ROOT.TColor.GetColor("#000099"),
}

legends = {
    "data": "Data",
    "Zgamma": "Z(#nu#nu)+#gamma",
    "Wgamma": "W(#it{l}#nu)+#gamma",
    "data_eprobe": "e#rightarrow#gamma",
    "data_y": "jet#rightarrow#gamma",
    "gammajet_direct": "#gamma+jets direct",
}

# Histogram settings for mT
nbins = 40
xmin = 0.0
xmax = 300.0  # GeV

# ------------------------------------------------------------
#  Fake-rate / weight helpers (scalar versions of your getWeight)
# ------------------------------------------------------------

def weight_jetfake(abs_eta):
    """Weights for data24_y jet->γ (piecewise in |η|)."""
    if 0.0 < abs_eta <= 0.6:
        return 1.453
    if 0.6 < abs_eta <= 1.37:
        return 1.603
    if 1.37 < abs_eta <= 1.52:
        return 0.0
    if 1.52 < abs_eta <= 1.81:
        return 1.942
    if 1.81 < abs_eta <= 2.37:
        return 1.545
    return 0.0


def weight_eprobe_scale(pt):
    """Scale(pt) part of e->γ weights (pt in MeV)."""
    if 50000 < pt <= 52000:
        return 1.318
    if 52000 < pt <= 54000:
        return 1.103
    if 54000 < pt <= 56000:
        return 1.118
    if 56000 < pt <= 58000:
        return 1.134
    if 58000 < pt <= 60000:
        return 1.107
    if 60000 < pt <= 62000:
        return 1.129
    if 62000 < pt <= 65000:
        return 1.137
    if 65000 < pt <= 70000:
        return 1.105
    if 70000 < pt <= 80000:
        return 1.125
    if 80000 < pt <= 100000:
        return 1.123
    if 100000 < pt <= 200000:
        return 1.109
    return 0.0


def weight_eprobe_norm(abs_eta):
    """norm(η) part of e->γ weights."""
    if 0.00 < abs_eta <= 0.60:
        return 0.03734
    if 0.60 < abs_eta <= 1.37:
        return 0.02868
    if 1.37 < abs_eta <= 1.52:
        return 0.0
    if 1.52 < abs_eta <= 1.81:
        return 0.04860
    if 1.81 < abs_eta <= 2.37:
        return 0.09252
    return 0.0


def weight_mc(entry):
    """
    MC weight for mc23e_* samples:
    mconly_weight/mc_weight_sum * xsec * filter_eff * kfactor * pu * jvt * 1000 * lumi
    """
    lumi = 109000.0  # mc23e
    return (
        entry.mconly_weight / entry.mc_weight_sum
        * entry.xsec_ami
        * entry.filter_eff_ami
        * entry.kfactor_ami
        * entry.pu_weight
        * entry.jvt_weight
        * 1000.0
        * lumi
    )

# ------------------------------------------------------------
#  Event selection (translation of your awk/np cuts)
# ------------------------------------------------------------

def passes_common_cuts(entry, pt_gamma, eta_gamma, phi_gamma):
    """Cuts common to all samples after iso / promotion."""
    # n_bjet == 0
    if entry.n_bjet != 0:
        return False

    # at least one "photon" (here we already have pt_gamma, eta_gamma from first object)
    if pt_gamma < 50000.0:   # ph_pt >= 50 GeV
        return False

    # lepton veto
    if entry.n_mu_baseline != 0:
        return False
    if entry.n_tau_baseline != 0:
        return False

    # photon multiplicity / electrons (for SR/MC/jetfake)
    # (for eprobe we ensured 1 electron and 0 photons earlier)
    # trigger
    if entry.trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M != 1:
        return False

    # MET
    if entry.met_tst_et < 100000.0:
        return False

    # n_jet_central <= 3
    if entry.n_jet_central > 3:
        return False

    # mT > 80 GeV (we compute mT in the caller)
    # met significance: 2 < met_tst_sig < 4
    if not (2.0 < entry.met_tst_sig < 4.0):
        return False

    # dphi_met_jetterm cleaning + cut
    dphi_j = entry.dphi_met_jetterm
    if dphi_j == -10.0:
        dphi_j = -999.0
    if dphi_j >= 0.75:
        return False

    # |eta_gamma| < 1.75
    if abs(eta_gamma) >= 1.75:
        return False

    # wrap Δφ(MET,γ) into [0, π] and apply > 0.5
    dphi_ph = entry.dphi_met_phterm
    dphi_wrapped = math.acos(math.cos(dphi_ph))
    if dphi_wrapped <= 0.5:
        return False

    return True


def compute_mt(met_et, met_phi, pt_gamma, phi_gamma):
    """mT (GeV) as in your notebook."""
    return math.sqrt(
        2.0 * met_et * pt_gamma * (1.0 - math.cos(met_phi - phi_gamma))
    ) / 1000.0


# ------------------------------------------------------------
#  Fill histograms for each sample
# ------------------------------------------------------------

hists = {}

for name in samples:
    cfg = sample_cfg[name]
    filename = f"{base_path}/{cfg['file']}_nominal_bdt.root"
    tree_name = cfg["tree"]
    stype = cfg["type"]

    print(f"Processing {name} from {filename} ({tree_name})")

    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {filename}")
    t = f.Get(tree_name)
    if not t:
        raise RuntimeError(f"Cannot get tree {tree_name} in {filename}")

    h = ROOT.TH1F(f"h_{name}", "", nbins, xmin, xmax)
    h.Sumw2()

    # Event loop
    nentries = t.GetEntries()
    for i in range(nentries):
        t.GetEntry(i)

        # ------------------------------------------------
        # Build "photon" kinematics and sample-specific pre-cuts
        # ------------------------------------------------
        if stype in ("data_sr", "mc", "jetfake"):
            # require at least one photon
            if t.ph_pt.size() == 0 or t.ph_eta.size() == 0 or t.ph_phi.size() == 0:
                continue

            pt_gamma = t.ph_pt[0]
            eta_gamma = t.ph_eta[0]
            phi_gamma = t.ph_phi[0]

            # electron veto for those samples
            if t.n_el_baseline != 0:
                continue

            # isolation region
            topo40 = t.ph_topoetcone40[0]
            iso = (topo40 - 2450.0) / pt_gamma

            if stype in ("data_sr", "mc"):   # tight iso
                if iso > 0.022:
                    continue
            elif stype == "jetfake":         # jet-fake control region
                if not (0.1 < iso < 0.4):
                    continue

        elif stype == "eprobe":
            # single-electron trigger & selection
            if t.trigger_single_el != 1:
                continue
            if t.n_el != 1:
                continue
            if t.n_ph_baseline != 0:
                continue

            if t.el_pt.size() == 0 or t.el_eta.size() == 0 or t.el_phi.size() == 0:
                continue

            pt_gamma = t.el_pt[0]
            eta_gamma = t.el_eta[0]
            phi_gamma = t.el_phi[0]

            # promote electron to "photon": Δφ(MET,γ) := Δφ(MET,e)
            # (we just reuse dphi_met_phterm branch as dphi_met_eleterm was used in the uproot code)
            # In PyROOT, this is already in dphi_met_eleterm.
            t.dphi_met_phterm = t.dphi_met_eleterm

        else:
            continue

        # ------------------------------------------------
        # Apply common cuts
        # ------------------------------------------------
        if not passes_common_cuts(t, pt_gamma, eta_gamma, phi_gamma):
            continue

        # ------------------------------------------------
        # Compute mT
        # ------------------------------------------------
        mt = compute_mt(t.met_tst_et, t.met_tst_phi, pt_gamma, phi_gamma)
        if mt <= 80.0:   # explicit mT cut as in notebook
            continue

        # ------------------------------------------------
        # Event weight
        # ------------------------------------------------
        if name == "data":
            w = 1.0
        elif stype == "mc":
            w = weight_mc(t)
        elif stype == "jetfake":
            w = weight_jetfake(abs(eta_gamma))
        elif stype == "eprobe":
            scale = weight_eprobe_scale(pt_gamma)
            norm = weight_eprobe_norm(abs(eta_gamma))
            w = scale * norm
        else:
            w = 1.0

        if w == 0.0:
            continue

        # Fill histogram
        h.Fill(mt, w)

    hists[name] = h
    f.Close()

# ------------------------------------------------------------
#  Build stack & ratio
# ------------------------------------------------------------

# Order for stacking (bottom -> top)
bkg_order = ["Zgamma", "Wgamma", "data_eprobe", "data_y", "gammajet_direct"]
data_hist = hists["data"]

stack = ROOT.THStack("hs", "")

# also build total MC histogram for ratio
h_mc_tot = None
for name in bkg_order:
    h = hists[name]
    color = colors[name]
    h.SetFillColor(color)
    h.SetLineColor(ROOT.kBlack)
    stack.Add(h, "hist")

    if h_mc_tot is None:
        h_mc_tot = h.Clone("h_mc_tot")
    else:
        h_mc_tot.Add(h)

# data style
data_hist.SetLineColor(colors["data"])
data_hist.SetMarkerColor(colors["data"])
data_hist.SetMarkerStyle(20)
data_hist.SetMarkerSize(1.0)

# ratio = data / MC
h_ratio = data_hist.Clone("h_ratio")
h_ratio.Divide(h_mc_tot)

# ------------------------------------------------------------
#  Canvas & pads
# ------------------------------------------------------------

c = ROOT.TCanvas("c", "", 800, 800)

pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.3)

pad1.SetBottomMargin(0.02)
pad2.SetTopMargin(0.05)
pad2.SetBottomMargin(0.3)

pad1.Draw()
pad2.Draw()

# ---- top pad: stacked plot ----
pad1.cd()
pad1.SetLogy(True)

stack.Draw("hist")
stack.GetXaxis().SetRangeUser(xmin, xmax)
stack.GetXaxis().SetLabelSize(0)  # hide x-axis labels on top pad
stack.GetYaxis().SetTitle("Events")
stack.GetYaxis().SetTitleSize(0.05)
stack.GetYaxis().SetTitleOffset(1.2)
stack.GetYaxis().SetLabelSize(0.04)
stack.SetMinimum(1e-4)
stack.SetMaximum(1e11)

data_hist.Draw("E1 same")

# legend
leg = ROOT.TLegend(0.55, 0.6, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)

for name in bkg_order:
    leg.AddEntry(hists[name], legends[name], "f")
leg.AddEntry(data_hist, legends["data"], "lep")
leg.Draw()

# ATLAS Simulation Internal label
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextFont(72)
latex.SetTextSize(0.05)
latex.DrawLatex(0.18, 0.88, "ATLAS")
latex2 = ROOT.TLatex()
latex2.SetNDC()
latex2.SetTextFont(42)
latex2.SetTextSize(0.05)
latex2.DrawLatex(0.32, 0.88, "Simulation Internal")

# ---- bottom pad: ratio ----
pad2.cd()

h_ratio.SetTitle("")
h_ratio.GetYaxis().SetTitle("data/MC")
h_ratio.GetYaxis().SetNdivisions(505)
h_ratio.GetYaxis().SetTitleSize(0.09)
h_ratio.GetYaxis().SetTitleOffset(0.5)
h_ratio.GetYaxis().SetLabelSize(0.08)

h_ratio.GetXaxis().SetTitle("m_{T}  [GeV]")
h_ratio.GetXaxis().SetTitleSize(0.1)
h_ratio.GetXaxis().SetTitleOffset(1.0)
h_ratio.GetXaxis().SetLabelSize(0.08)

h_ratio.SetMinimum(0.0)
h_ratio.SetMaximum(2.0)

h_ratio.Draw("E1")

# horizontal lines at 1, 0.5, 1.5
line1 = ROOT.TLine(xmin, 1.0, xmax, 1.0)
line1.SetLineColor(ROOT.kBlack)
line1.SetLineWidth(1)
line1.Draw("same")

line05 = ROOT.TLine(xmin, 0.5, xmax, 0.5)
line05.SetLineColor(ROOT.kGray+1)
line05.SetLineStyle(2)
line05.Draw("same")

line15 = ROOT.TLine(xmin, 1.5, xmax, 1.5)
line15.SetLineColor(ROOT.kGray+1)
line15.SetLineStyle(2)
line15.Draw("same")

pad2.SetGridy(True)

c.cd()
c.Update()
c.SaveAs("validation_mt_pyroot.png")

print("Saved plot as validation_mt_pyroot.png")
