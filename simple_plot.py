import os, ROOT, math, re, sys, argparse
import numpy as np

ROOT.gROOT.SetStyle("ATLAS")
ROOT.gROOT.SetBatch(True)

ROOT.gStyle.SetOptStat(0)

def getSelDict():
    dict= {}
    dict['met100phPT50'] = {'str' : 'met_tst_et > 100000 && trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M && ph_pt[0]>50000'}
    return dict
sel_dict = getSelDict()

def getVarDict():
    dict = {}
    dict['vtx_sumPt'] = {'var':'vtx_sumPt','bins':[20,0,100], 'title': 'vtx_sumPt', 'shift':'+0'}
    return dict
var_dict = getVarDict()


def getSampleDict():
    dict = {}
    dict['Zjets'] =             {'color': ROOT.kGreen-2,    'legend': 'Z(#nu#nu,ll)+jets',                  'tree': 'nominal',   'filenames': ['Zjets']}
    dict['Zgamma'] =            {'color': ROOT.kOrange+7,   'legend': 'Z(#nu#nu)+#gamma',                   'tree': 'nominal',   'filenames': ['Zgamma']}
    dict['Wgamma'] =            {'color': ROOT.kOrange+1,   'legend': 'W(l#nu)+#gamma',                     'tree': 'nominal',   'filenames': ['Wgamma']}
    dict['Wjets'] =             {'color': ROOT.kTeal+5,     'legend': 'W(l#nu)+jets',                       'tree': 'nominal',   'filenames': ['Wjets']}
    dict['gammajet_direct'] =   {'color': ROOT.kBlue+2,     'legend': '#gamma+jets direct',                 'tree': 'gammajets', 'filenames': ['gammajet_direct']}
    dict['gammajet_frag'] =     {'color': ROOT.kBlue-5,     'legend': '#gamma+jets frag',                   'tree': 'gammajets', 'filenames': ['gammajet_frag']}
    dict['dijet'] =             {'color': ROOT.kCyan+1,     'legend': 'multijets',                          'tree': 'dijets',    'filenames': ['dijet']}
    dict['ggHyyd'] =            {'color': ROOT.kRed,        'legend': 'ggH, H#rightarrow#gamma#gamma_{d}',  'tree': 'nominal',   'filenames': ['ggHyyd']}
    return dict
sample_dict = getSampleDict()

dir = "/data/tmathew/ntups/mc23d"
samples = ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct','gammajet_frag','dijet']
sels = ["met100phPT50"]
year = "mc23d"
stack = True
norm = False
rnorm = False
rsig = True
data = False
doratio = True
blind = False
dosigGoodPV = False
roc = True
tag = ""


dict = {}
vars = var_dict

def getWeightDict():
    dict = {}
    dict['ggHyyd'] = 140.587904214859
    dict['Zjets'] = 2155.7233924865723
    dict['Zgamma'] = 11350.253204345703
    dict['Wgamma'] = 8653.602767944336
    dict['Wjets'] = 12703.016357421875
    dict['gammajet_direct'] = 218518.66864013672
    dict['gammajet_frag'] = 161092.83154296875
    dict['dijet'] = 171501971.1118164
    return dict
weight_dict = getWeightDict()

def GetSelString(sel, addcut = ''):
    cuts_string = []
    if sel in sel_dict: cuts_string.append(sel_dict[sel]['str'])
    else: 
        cuts = sel.split('_')
        for cut in cuts: 
            cuts_string.append(GetCutString(cut))
    if addcut != '': cuts_string.append(addcut)
    selstr = " && ".join(cuts_string)
    return selstr

def GetCutString(cut):
    if cut not in sel_dict: 
        c = re.split("[>,<,>=,<=,==,!=]", cut)
        var = c[0]
        cut = cut.replace(var, var_dict[var]['var'])
    else: cut = sel_dict[cut]['str']
    return cut

def removeCut(var, totsel):
    allcuts = []
    if var['var'] in totsel: 
        cuts = totsel.split('&&')
        for c in cuts:
            if var['var'] not in c: allcuts.append(c)
        totsel = '&&'.join(allcuts)
    return 0 

def getLumi(period):
    if period == "mc23d" in period: return '25767.5'
    if period == 'Run2' : return '((year<=2016)*36640+(year==2017)*44630 +(year==2018)*58790)'
def GetLegend(Ncol = 2):
    ROOT.gStyle.SetLegendBorderSize(0)
    legend = ROOT.TLegend(0.5,0.7,0.88,0.88)
    legend.SetTextSize(0.04)
    legend.SetTextFont(42)
    legend.SetNColumns(Ncol)
    return legend

def CreateCanvas(doratio,logy=True):
    canv = ROOT.TCanvas("c","c",600,600)
    canv.cd()
    pad1 = ROOT.TPad("pad1","pad1",0,0.35,1,1)
    pad1.SetNumber(1)
    pad1.SetBottomMargin(0.005)
    pad2 = ROOT.TPad("pad2","pad2",0,0,1,0.35)
    pad2.SetBottomMargin(1./3.)
    pad2.SetNumber(2)
    pad1.Draw()
    pad2.Draw()
    pad1 = canv.cd(1)
    if logy: pad1.SetLogy()
    pad2 = canv.cd(2)
    return canv

def fillChain(chain, file_path, period, sample, sel = 'y'):
    for file in sample_dict[sample]['filenames']: chain.Add('%s/%s_%s.root' %(file_path, file.replace('_j','').replace('_e',''), sel))
    return chain

def Plot(canv, Legend, file_path, h, sample, var, sel, counter = 1, normalize = False, period = 'mc23d', logy = True, rebin = False, stack = None, markersize = 0, markerstyle = 0, linestyle=0, ratio = True, col = '', leg = '', sigGoodPV = False, reweight = '', plot=True):
 
    finalState = 'y'
    samples_dict = getSampleDict()

    chain = ROOT.TChain(samples_dict[sample]['tree'] if period == 'mc23c' else 'nominal')
    fillChain(chain, file_path, period, sample, sel = finalState)
    # total_events = chain.GetEntries()
    # print("Sample {}: Total events before cuts = {}".format(sample, total_events))

    weight = weight_dict[sample]
    doStack = (stack != None and sample not in ['ggHyyd','data'])

    print(sel)
    print(weight)

    if sample == 'ggHyyd' and sigGoodPV: weight = weight+'*goodPV' #to show only signal with good PV

    # If no col and leg explicitely set, will assume you're comparing distributions for different samples
    color = samples_dict[sample]['color'] if col == '' else col
    legend = samples_dict[sample]['legend'] if leg == '' else leg
    shift = '*1'#var_dict[var]['shift'] if 'met_tst_et > 50000' in sel else '*1' #if var == 'mt' else '*1'

    #varstr = var_dict[var]['var'] if var!='mt' or period!='mc23d' else 'mt/1000'
    varstr = var_dict[var]['var'] if var!='mt' or period!='mc23d' else 'sqrt(2*met_tst_et*ph_pt[0]*(1-cos(met_tst_phi-ph_phi[0])))/1000'

    
    h.SetLineColor(color)
    if sample != 'data':
        if (doStack): 
            h.SetFillColor(color)
            h.SetLineColor(ROOT.kBlack)
        h.SetLineStyle(linestyle)
        h.SetMarkerSize(markersize)
        h.SetMarkerStyle(markerstyle)
        h.SetMarkerColor(color)

    ctmp = ROOT.TCanvas()
    ctmp.cd()

    if '_j' in sample: sel = sel + '&& ph_truth_origin[0]!=12 &&  ph_truth_origin[0]!=13'
    if '_e' in sample: sel = sel + '&& (ph_truth_origin[0]==12 || ph_truth_origin[0]==13)'
    chain.Draw('(%s%s) >> %s' %(varstr,shift ,h.GetName()), '(%s) * (%s)' %(sel,weight))

    # events_after_cuts = h.Integral()
    # print("Sample {}: Events after cuts = {}".format(sample, events_after_cuts))
    
    canv.cd(1) if ratio else canv.cd()
    h.GetYaxis().SetLabelSize(0.05)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetXaxis().SetTitle(var_dict[var]['title'])
    h.GetXaxis().SetLabelSize(0.05)
    h.GetXaxis().SetTitleSize(0.05)

    if rebin : h.Rebin(2)

    if stack == None and plot: 
        if not normalize: 
            h.GetYaxis().SetRangeUser(0.011,100000000)
            h.GetYaxis().SetTitle('Events')
            h.Draw('sameE')
        else: 
            h.GetYaxis().SetRangeUser(0.1,h.GetMaximum()*100)
            h.DrawNormalized('PE' if counter == 0 else 'sameE')

    if doStack: 
        h.GetYaxis().SetRangeUser(0.00000011,100000000000)
        h.SetMaximum(100000000000)
        h.GetYaxis().SetTitleOffset(1)
        h.GetYaxis().SetTitle('Events')
        h.GetYaxis().SetTitleSize(0.05)
        h.GetXaxis().SetTitle(var_dict[var]['title'])
        h.GetXaxis().SetLabelSize(0.05)
        h.GetXaxis().SetTitleSize(0.05)
        if plot: stack.Add(h)
    if sample != 'subtract': Legend.AddEntry(h,legend,'F' if (doStack or (stack==None and sample == 'fakeMET')) else 'L')
    canv.Update()
    print(sample, h.Integral())

    return h

def PlotRatio(canv, h, h0, ratio, var, color, normalize = False, rebin = False, significance = False, refline = None):
    canv.cd(2)

    # Plot h/h0 in ratio plot
    if significance == False: 
        hcopy = h.Clone()
        h0copy = h0.Clone()
        if normalize:
            if hcopy.Integral()>0: hcopy.Scale(1./hcopy.Integral())
            if h0copy.Integral()>0: h0copy.Scale(1./h0copy.Integral())
        ratio.Add(hcopy)
        if h0copy.Integral()>0: ratio.Divide(h0copy)
        else: ratio.Scale(0)

    # Plot significance in ratio plot
    else:
        if rebin: ratio.Rebin(2)
        bins = h.GetNbinsX()
        for b in range(bins):
            #print('signal events', h.Integral())
            if h0.GetBinContent(b+1)>0: ratio.SetBinContent(b+1, h.GetBinContent(b+1)/math.sqrt(h0.GetBinContent(b+1)))
            else: ratio.SetBinContent(b+1,0)

    ratio.SetLineColor(color)
    ratio.SetMarkerSize(0)
    ratio.SetMarkerColor(color)
    ratio.GetYaxis().SetTitle('data/MC' if not significance else 's/#sqrt{b}')
    ratio.GetYaxis().SetTitleSize(0.05*65./35.)
    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleOffset(1/65.*35.)
    ratio.GetYaxis().SetRangeUser(0,1)
    if significance==True : ratio.GetYaxis().SetRangeUser(0,1)
    #ratio.Draw('PE' if counter == 0 else 'samePE')
    ratio.GetXaxis().SetTitle(var_dict[var]['title'])
    ratio.GetXaxis().SetLabelSize(0.05*65./35.)
    ratio.GetXaxis().SetTitleSize(0.05*65./35.)
    ratio.GetYaxis().SetLabelSize(0.05*65./35.)
    ratio.GetYaxis().SetTitleSize(0.05*65./35.)
    ratio.Draw('same')
    if refline != None:
        print('draw line')
        refline.SetLineStyle(2)
        refline.SetLineColor(ROOT.kBlack)
        refline.SetLineWidth(1)
        refline.Draw()

    # Calculate and display overall significance
    total_signal = h.Integral()
    total_background = h0.Integral()
    if total_background > 0:
        overall_significance = total_signal / math.sqrt(total_background)
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.08)
        latex.SetTextColor(ROOT.kViolet)
        latex.DrawLatex(0.72, 0.8, f"S/#sqrt{{B}} = {overall_significance:.5f}")
    else:
        print("Warning: Total background is zero, significance cannot be calculated.")

    canv.Update()
    return ratio

def PlotROCStandalone(sig_hist, bkg_hist, output_filename="tmp/ROC_curve.png"):
    # Check if histograms are empty
    if sig_hist.Integral() == 0 or bkg_hist.Integral() == 0:
        print("Warning: Empty signal or background histogram. Skipping ROC plot.")
        return  

    bins = sig_hist.GetNbinsX()
    sig_eff = []
    bkg_rej = []

    for i in range(1, bins + 1):
        signal_integral = sig_hist.Integral(i, bins)
        background_integral = bkg_hist.Integral(i, bins)

        signal_efficiency = signal_integral / sig_hist.Integral() if sig_hist.Integral() > 0 else 0
        background_efficiency = background_integral / bkg_hist.Integral() if bkg_hist.Integral() > 0 else 0
        background_rejection = 1 - background_efficiency

        sig_eff.append(signal_efficiency)
        bkg_rej.append(background_rejection)

    sig_eff = np.array(sig_eff, dtype=float)
    bkg_rej = np.array(bkg_rej, dtype=float)

    # values are sorted from (0,0) to (1,1)
    sorted_indices = np.argsort(sig_eff)
    sig_eff = sig_eff[sorted_indices]
    bkg_rej = bkg_rej[sorted_indices]

    auc = np.trapz(bkg_rej, sig_eff)  # Integrate using the trapezoidal rule
    print(f"AUC (Area Under Curve) = {auc:.4f}")

    # Create the ROC curve graph
    roc_graph = ROOT.TGraph(len(sig_eff), sig_eff, bkg_rej)
    roc_graph.SetTitle("ROC Curve;Signal Efficiency;Background Rejection")
    roc_graph.SetLineColor(ROOT.kViolet)
    roc_graph.SetLineWidth(2)
    roc_graph.SetMarkerStyle(20)
    roc_graph.SetMarkerSize(0.5)

    # Debugging: Print the number of points
    print("ROC Graph points:", roc_graph.GetN())

    # Ensure X and Y axes are correctly scaled
    roc_graph.GetXaxis().SetLimits(0, 1)  # Ensure X is [0,1]
    roc_graph.GetYaxis().SetRangeUser(0, 1)  # Ensure Y is [0,1]
    roc_graph.GetXaxis().SetTitleSize(0.05)
    roc_graph.GetYaxis().SetTitleSize(0.05)
    roc_graph.GetXaxis().SetLabelSize(0.04)
    roc_graph.GetYaxis().SetLabelSize(0.04)

    # Print ROC graph points for debugging
    roc_graph.Print()

    # Create standalone canvas
    canv = ROOT.TCanvas("roc_canvas", "ROC Curve", 600, 600)
    canv.SetLeftMargin(0.15)
    canv.SetRightMargin(0.05)
    canv.SetTopMargin(0.05)
    canv.SetBottomMargin(0.15)

    # Draw the ROC Curve
    roc_graph.Draw("APL")  # A: Axis, P: Points, L: Line

    # Add a legend
    legend = ROOT.TLegend(0.65, 0.8, 0.85, 1.0)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.03)
    legend.AddEntry(roc_graph, f"AUC = {auc:.4f}", "l")
    legend.Draw()

    # Add a diagonal line from (0, 1) to (1, 0)
    diag_line = ROOT.TLine(0, 1, 1, 0)
    diag_line.SetLineStyle(2)
    diag_line.SetLineColor(ROOT.kRed)
    diag_line.Draw()

    canv.SetGrid()

    # Save the canvas as an image
    canv.SaveAs(output_filename)
    print(f"ROC curve saved as {output_filename}")

    return roc_graph

def PlotSignificanceVsCut(variable, start, stop, step, base_sel, file_path, period, samples):
    thresholds, significances = [], []
    
    # Loop over the thresholds
    for thresh in range(start, stop + step, step):
        # Construct a new selection: base_sel && (variable > threshold)
        # (If you want the cut to be ">" rather than ">=" adjust accordingly.)
        cut_sel = base_sel + " && " + variable + " > " + str(thresh)
        print(f"Applying selection: {cut_sel}")
        
        sig_yield = 0.0
        bkg_yield = 0.0
        
        # Loop over the samples
        for sample in samples:
            chain = ROOT.TChain("nominal")
            fillChain(chain, file_path, period, sample)
            
            weight = weight_dict[sample]
            n_entries = chain.GetEntries(cut_sel)
            
            if sample == "ggHyyd":
                sig_yield += n_entries * weight
            else:
                bkg_yield += n_entries * weight
        
        # Calculate significance as S/sqrt(B) (if B > 0)
        if bkg_yield > 0:
            significance = sig_yield / (bkg_yield ** 0.5)
        else:
            significance = 0.0
        
        thresholds.append(thresh)
        significances.append(significance)
        
        print(f"Threshold: {thresh}, Signal yield: {sig_yield:.1f}, Background yield: {bkg_yield:.1f}, Significance: {significance:.3f}")
    
    # Create a TGraph with the results.
    npoints = len(thresholds)
    graph = ROOT.TGraph(npoints)
    for i in range(npoints):
        graph.SetPoint(i, thresholds[i], significances[i])
    
    # Set the graph title and axis labels.
    graph.SetTitle("Significance vs. Cut on " + variable + ";" + variable + " cut threshold;Significance S/#sqrt{B}")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1)
    
    return graph

# # --- Create Signficance vs. cut --- #
# # base_selection = "n_ph == 1 && n_ph_baseline == 1 && n_tau_baseline == 0 && n_mu_baseline == 0 && n_el_baseline == 0 && trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M && met_tst_sig > 6 && abs(ph_eta[0]) < 1.75 && (met_tst_noJVT_et - met_tst_et) > -10000 && ((met_jetterm_et != 0)*Alt$(acos(cos(met_tst_phi - met_jetterm_phi)),0) + (met_jetterm_et == 0)*0) < 1.5"
# base_selection = 'n_ph == 1 && n_ph_baseline == 1 && n_tau_baseline == 0 && n_mu_baseline == 0 && n_el_baseline == 0 && trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M'
# file_path = "/data/tmathew/ntups/mc23d"
# period = "mc23d"
# samples = ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct','gammajet_frag','dijet']

# roc_graph_met = PlotSignificanceVsCut("met_tst_et", 0, 100000, 10000, base_selection, file_path, period, samples)

# # Plot significance vs ph_pt cut:
# roc_graph_ph = PlotSignificanceVsCut("ph_pt[0]", 0, 80000, 10000, base_selection, file_path, period, samples)

# # Now, create a canvas to draw one of the graphs.
# c = ROOT.TCanvas("c", "Significance vs. Cut", 650, 600)
# roc_graph_met.Draw("AP")  # "A" for axis, "P" for points.
# c.SaveAs("tmp/Significance_vs_met_tst_et_cut.png")

# c2 = ROOT.TCanvas("c2", "Significance vs. Cut", 650, 600)
# roc_graph_ph.Draw("AP")
# c2.SaveAs("tmp/Significance_vs_ph_pt_cut.png")

if True:
    for var in vars:     # Loop over variables 
        for sel in sels: # Loop over selections

            nbin,minbin,maxbin = var_dict[var]['bins'][0],var_dict[var]['bins'][1],var_dict[var]['bins'][2]
            totsel = GetSelString(sel)
            
            print(removeCut(var_dict[var], totsel)) # Remove cut on plotted variables

            print ('Plotting variable %s in region %s ... ' %(var, totsel))
            lumi = getLumi(year)
            Legend = GetLegend()
            canv = CreateCanvas(doratio)

            h = []                                                  # Vector of histograms, one for each sample
            hbkg = ROOT.TH1F('hbkg','hbkg',nbin,minbin,maxbin)      # Sum of background histograms
            hsig = ROOT.TH1F('hsig','hsig',nbin,minbin,maxbin)      # Signal histogram
            hdata = ROOT.TH1F('hdata','hdata',nbin,minbin,maxbin)   # Data histogram    # Signal histogram
            

            # --- Create THStack --- #
            hs = ROOT.THStack("hs","") 
            hs.SetMinimum(0.001)
            hs.SetMaximum(100000000000)

            canv.cd()

            # --- Fill and draw one histogram per sample --- #
            for i,sample in enumerate(samples): 
                h.append(ROOT.TH1F('h%i' %i,'h%i' %i,nbin,minbin,maxbin))
                Plot(canv, Legend, dir, h[i], sample, '%s' %var, totsel, i, period = year, normalize=norm, stack = hs, ratio=doratio, sigGoodPV = dosigGoodPV)
                if sample != 'ggHyyd' and sample != 'data': hbkg.Add(h[i])
                if sample == 'ggHyyd': hsig.Add(h[i])
                if sample == 'data': hdata.Add(h[i])
                if var == 'vertex': print (sample, h[i].GetBinContent(2)/h[i].Integral())
                if var == 'goodPV': print(sample,(h[i].GetBinContent(2)/h[i].Integral()))


            hbkg.SetLineColor(ROOT.kBlack)
            hbkg.SetMarkerSize(0)
            hbkg.SetFillStyle(3004)
            hbkg.SetFillColor(ROOT.kBlack)
            hsig.SetLineColor(ROOT.kRed)
            hsig.SetMarkerSize(0)
            print("total bkg: " , hbkg.Integral())
            
            hs.Draw('HIST')
            hs.GetXaxis().SetTitle(var_dict[var]['title'])
            hs.GetXaxis().SetLabelSize(0.05)
            hs.GetXaxis().SetTitleSize(0.05)
            hs.GetYaxis().SetTitleOffset(1.2)
            hs.GetYaxis().SetTitle('Events')
            hs.GetYaxis().SetTitleSize(0.05)
            hbkg.Draw('sameE2')
            hsig.Draw('sameHISTE')
            hdata.Draw('sameE2')

            ratio = []  # Vector or ratio histograms, one per sample
            canv.cd(2)
            ratio.append(ROOT.TH1F('ratio','ratio',nbin,minbin,maxbin))
            PlotRatio(canv, hsig, hbkg, ratio[0], '%s' %var,  ROOT.kBlack,normalize=rnorm, significance=rsig)
            if roc:
                # print("it is drawing roc curve")
                PlotROCStandalone(hsig, hbkg)                

            canv.cd(1)

            Legend.Draw()

            canv.SaveAs('tmp/%s_%s%s_%s.png' %(var,sel, 'Data' if data else '', tag))

            
