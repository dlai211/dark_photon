import os
import ROOT
import math
import re
import sys
import numpy as np

import argparse

sys.path.append('Dictionaries.py')


import Dictionaries
from Dictionaries import *

ROOT.gROOT.SetStyle("ATLAS")
ROOT.gStyle.SetOptStat(0)


sel_dict = getSelDict()
var_dict = getVarDict()
sample_dict = getSampleDict()

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
        print(var)
        cut = cut.replace(var, var_dict[var]['var'])
    else: cut = sel_dict[cut]['str']
    return cut

def GetBinning(var):
    nbin,minbin,maxbin = var_dict[var]['bins'][0],var_dict[var]['bins'][1],var_dict[var]['bins'][2]
    return nbin,minbin,maxbin

def getLumi(period):
    if period == "mc23d" in period: return '25767.5'
    if period == 'Run2' : return '((year<=2016)*36640+(year==2017)*44630 +(year==2018)*58790)'

def getWeight(period, sample):
    lumi = getLumi(period)
    weight = '(mconly_weight/mc_weight_sum)*xsec_ami*filter_eff_ami*kfactor_ami*pu_weight*jvt_weight*1000*%s' %(lumi)
    if sample == 'data' or sample == 'fakeMET' or sample == 'jetFake': weight='1'
    if sample in ['ggHyyd','WH','VBF','ZH'] : 
        xsec_sig = '0.052' if ( period == 'Run3' or 'mc23' in period ) else '0.048'
        if sample != 'ggHyyd' : xsec_sig = 'xsec_ami'
        br = '0.01'
        weight = '(mconly_weight/mc_weight_sum)*%s*pu_weight*jvt_weight*filter_eff_ami*kfactor_ami*%s*1000*%s' %(xsec_sig,lumi,br)
    return weight

def removeCut(var, totsel):
    allcuts = []
    if var['var'] in totsel: 
        cuts = totsel.split('&&')
        for c in cuts:
            if var['var'] not in c: allcuts.append(c)
        totsel = '&&'.join(allcuts)
    return 0 

def fillChain(chain, file_path, period, sample, sel = 'y'):
    for file in sample_dict[sample]['filenames']: chain.Add('%s/%s_%s.root' %(file_path, file.replace('_j','').replace('_e',''), sel))
    return chain

def getFinalState(sel):
    if ('n_mu == 1' or 'n_mu==1') in sel: return 'uy'
    elif ('n_mu == 2' or 'n_mu==2') in sel: return 'uuy'
    else: return 'y'

#### PLOT SETTINGS ####

def GetLegend(Ncol = 2):
    ROOT.gStyle.SetLegendBorderSize(0)
    legend = ROOT.TLegend(0.5,0.7,0.88,0.88)
    legend.SetTextSize(0.04)
    legend.SetTextFont(42)
    legend.SetNColumns(Ncol)
    return legend

def GetAtlasLabel():
    atlas_label = ROOT.TLatex(0.50, 0.89, "#bf{#it{ATLAS}} Simulation Internal")
    atlas_label.SetNDC()
    atlas_label.SetTextSize(0.04)
    return atlas_label

def CreateCanvas(doratio,logy=True):

    canv = ROOT.TCanvas("c","c",600,600)

    if doratio:
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

    else:
        canv = ROOT.TCanvas("c","c",800,600)
        if logy: canv.SetLogy()
        canv.cd()

    return canv


##### PLOT FUNCTIONS #####

def Plot(canv, Legend, file_path, h, sample, var, sel, counter = 1, normalize = False, period = 'mc23d', logy = True, rebin = False, stack = None, markersize = 0, markerstyle = 0, linestyle=0, ratio = True, col = '', leg = '', sigGoodPV = False, reweight = '', plot=True):
 
    finalState = getFinalState(sel)
    samples_dict = getSampleDict()

    chain = ROOT.TChain(samples_dict[sample]['tree'] if period == 'mc23c' else 'nominal')
    fillChain(chain, file_path, period, sample, sel = finalState)
    # total_events = chain.GetEntries()
    # print("Sample {}: Total events before cuts = {}".format(sample, total_events))

    weight = getWeight(period, sample) if reweight == '' else '(%s*%s)' %(getWeight(period, sample),reweight)
    doStack = (stack != None and sample not in ['ggHyyd','data'])

    print('sel: ', sel)
    print('weight: ', weight)

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
    """
    Plots the ROC curve in a standalone canvas and saves it as an image.
    """

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
    """
    Loop over thresholds on 'variable' and compute S/sqrt(B) for each cut.
    
    Parameters:
      variable (str): The branch name on which to apply the cut (e.g. "met_tst_et" or "ph_pt").
      start (int): The minimum threshold value.
      stop (int): The maximum threshold value.
      step (int): The step size between thresholds.
      base_sel (str): The base selection string (in ROOT syntax) already applied.
                        (It should not include a condition on 'variable'.)
      file_path (str): Directory containing the ROOT files.
      period (str): The period string (e.g. "mc23d").
      samples (list): List of sample names (e.g. ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct','gammajet_frag','dijet']).
    
    Returns:
      TGraph: A ROOT.TGraph with the cut threshold on the x-axis and significance on the y-axis.
    """
    # Lists to store the threshold values and corresponding significance
    thresholds = []
    significances = []
    
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
            # Create a TChain for this sample.
            # Use the tree name "nominal" (or adjust if needed).
            chain = ROOT.TChain("nominal")
            # fillChain is your helper function from PlotUtilities
            fillChain(chain, file_path, period, sample)
            
            # Get the weight factor for this sample (as a string); convert it to a float.
            weight_str = getWeight(period, sample)
            try:
                weight = float(weight_str)
            except Exception:
                # For data or unweighted samples, set weight = 1.
                weight = 1.0

            # Get the number of entries passing the current cut selection.
            # (Note: This returns the raw number of events. Multiplying by the weight gives an approximate yield.)
            n_entries = chain.GetEntries(cut_sel)
            
            # Accumulate signal and background yields.
            if sample == "ggHyyd":
                sig_yield += n_entries * weight
            else:
                # Exclude data if present
                if sample != "data":
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


def Plot2D(canv,dir, h, sample, var, sel, period = 'mc23d', logy = True, sigGoodPV = False):
    file_path = dir
    chain = ROOT.TChain(samples_dict[sample]['tree'] if period!='MC21New' else sample)

    fillChain(chain, file_path, period, sample)


    weight = getWeight(period, sample)
    # weight = weight_dict[sample]
    if sample == 'ggHyyd' and sigGoodPV: weight = weight+'*goodPV'

    v1, v2 = var.split('VS')[0],var.split('VS')[1]
    vstr1, vstr2 = var_dict[v1]['var'],var_dict[v2]['var']
    
    #print(varstr, sample)

    ctmp = ROOT.TCanvas()
    ctmp.cd()
    #print('(%s) * (%s)' %(sel,weight))
    chain.Draw('%s:%s >> %s' %(vstr2,vstr1,h.GetName()), '(%s) * (%s)' %(sel,weight), "COLZ")
    canv.cd()
    h.GetYaxis().SetTitle(var_dict[v2]['title'])
    h.GetYaxis().SetLabelSize(0.05)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetXaxis().SetTitle(var_dict[v1]['title'])
    h.GetXaxis().SetLabelSize(0.05)
    h.GetXaxis().SetTitleSize(0.05)
    h.Draw("COLZ")


    canv.Update()

    return h

def Blind(hdata, hsig):
    Nbins = hdata.GetNbinsX()
    for b in range(Nbins-1):
        print('bin ', b)
        if hdata.GetBinContent(b+1)>0:
            if hsig.GetBinContent(b+1)/hdata.GetBinContent(b+1)>0.01: 
                print('signal %:', hsig.GetBinContent(b+1)/hdata.GetBinContent(b+1))
                hdata.SetBinContent(b+1,0)
                hdata.SetBinError(b+1,0)
            
    return 0



##### CUTFLOW #####

def Cutflow(dir, samples, sel, period):
    h = {}
    selpartial = ''
    cuts = sel.split('_')
    for i,cut in enumerate(cuts):
        h[cut] = {}
        bkg = 0
        sig = 0
        #print (i)
        selpartial =  selpartial + " && " + sel_dict[cut]['str'] if (i!=0) else sel_dict[cut]['str']

        for sample in samples:

            if '_j' in sample: selpartial = selpartial + '&& ph_truth_origin[0]!=12 &&  ph_truth_origin[0]!=13'
            if '_e' in sample: selpartial = selpartial + '&& (ph_truth_origin[0]==12 || ph_truth_origin[0]==13)'
            #print(selpartial)
            print(sample)
            weight = getWeight(period,sample)
            chain = ROOT.TChain(samples_dict[sample]['tree'] if period=='mc23c' else 'nominal')
            fillChain(chain, dir, period, sample)
            h[cut][sample] = ROOT.TH1F('h%s' %sample,'h%s' %sample,1,0,1)
            chain.Draw('0.5 >> %s' %(h[cut][sample].GetName()), '(%s)*(%s)' %(weight ,selpartial))
            if sample != 'ggHyyd' : bkg=bkg + h[cut][sample].Integral()
            else: sig = h[cut][sample].Integral()
            selpartial = selpartial.replace('&& ph_truth_origin[0]!=12 &&  ph_truth_origin[0]!=13','').replace('&& (ph_truth_origin[0]==12 || ph_truth_origin[0]==13)','')
        h[cut]['sig'] = sig/math.sqrt(bkg)
            
        string = ''
        for sample in samples:
            string += '\t '+str(h[cut][sample].Integral())+ '\t '+str(h[cut][sample].Integral()/h[cuts[0]][sample].Integral())
        string += '\t '+str(h[cut]['sig'])

        print (cut, string)



    lineHeader = "{:>50}"
    for sample in samples:
        lineHeader = "%s%s" %(lineHeader,'{:>15}')
    lineHeader = "%s%s" %(lineHeader,'{:>15}')

    line = "{:>50}"
    for sample in samples:
        line = "%s%s" %(line,'{:>15.4g}')
    line = "%s%s" %(line,'{:>15.4g}')

    rowHeader = ['']
    for sample in samples: rowHeader.append(sample)
    rowHeader.append('s/sqrt(b)')
    print(lineHeader.format(*rowHeader))

    for i,cut in enumerate(cuts):    
        row = [cut]
        for sample in samples:
            row.append(h[cut][sample].Integral())

        row.append(h[cut]['sig'])
        print(line.format(*row))




    
