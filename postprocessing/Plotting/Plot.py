import os
import ROOT

import argparse

ROOT.gROOT.SetStyle("ATLAS")
ROOT.gROOT.SetBatch(True)

import sys
sys.path.append('PlotUtilities.py')
sys.path.append('BkgEstimate.py')
import PlotUtilities
import BkgEstimate
from PlotUtilities import *
from BkgEstimate import *



parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--samples', type=str, nargs='+', help='list of samples', default=['all'])
parser.add_argument('-v', '--vars', type=str, nargs='+', help='list of variables', default=['all'])
parser.add_argument('-c', '--cuts', type=str, nargs='+', help='list of selections', default=['met100phPT50'])
parser.add_argument('--stack', action = 'store_true',  help='do stack plot', default=False)
parser.add_argument('-p', '--period', type=str, help='period (Run2,mc23d)', default='mc23d')
parser.add_argument('-d', '--dir', type=str, help='input directory', default='/data/tmathew/ntups/mc23d')
parser.add_argument('--norm', action = 'store_true',  help='normalize', default=False)
parser.add_argument('--rnorm', action = 'store_true',  help='normalize ratio plot', default=False)
parser.add_argument('--rsig', action = 'store_true',  help='plots significance in ratio plot', default=False)
parser.add_argument('--data', action = 'store_true',  help='show data', default=False)
parser.add_argument('--ratio', action = 'store_true',  help='include ratio plot', default=False)
parser.add_argument('--sigGoodPV', action = 'store_true',  help='plot only good vertex signals', default=False)
parser.add_argument('--blind', action = 'store_true',  help='blind', default=False)
parser.add_argument('--tag',  type=str, help='tag', default='')

ROOT.gStyle.SetOptStat(0)



if __name__ == '__main__':

    presel = "1==1" 
    

    ##### PARSER ARGUMENTS #####
    args=parser.parse_args()
    dir = args.dir
    vars = args.vars if args.vars!=['all'] else var_dict
    samples = args.samples if args.samples!=['all'] else ['ggHyyd','Zjets','Zgamma','Wgamma','Wjets','gammajet_direct','gammajet_frag','dijet']
    sels = args.cuts
    year = args.period
    stack = args.stack
    norm = args.norm
    rnorm = args.rnorm
    rsig = args.rsig
    data = args.data
    doratio = args.ratio
    blind = args.blind

    tag = args.tag
        
    #########################

    dict = {}

    var_dict = getVarDict()
    samples_dict = getSampleDict()

    for var in vars:     # Loop over variables 
        for sel in sels: # Loop over selections

            ###### INITIALIZATION ######

            nbin,minbin,maxbin = GetBinning(var)
            totsel = GetSelString(sel)
            
            removeCut(var_dict[var], totsel) # Remove cut on plotted variables

            print ('Plotting variable %s in region %s ... ' %(var, totsel))
            lumi = getLumi(year)
            Legend = GetLegend()
            atlas_label = GetAtlasLabel()
            canv = CreateCanvas(doratio)

            # --- Histograms --- #
            h = []                                                  # Vector of histograms, one for each sample
            hbkg = ROOT.TH1F('hbkg','hbkg',nbin,minbin,maxbin)      # Sum of background histograms
            hsig = ROOT.TH1F('hsig','hsig',nbin,minbin,maxbin)      # Signal histogram
            hdata = ROOT.TH1F('hdata','hdata',nbin,minbin,maxbin)   # Data histogram    # Signal histogram
            
            # --- Create THStack if needed --- #
            hs = None
            if stack: 
                hs = ROOT.THStack("hs","") 
                hs.SetMinimum(0.001)
                hs.SetMaximum(100000000000)


            ##### DRAW MAIN PAD #####

            canv.cd(1) if doratio else canv.cd()
            if data and 'data' not in samples: samples.append('data')
            print(samples)
            

            # --- Fill and draw one histogram per sample --- #
            for i,sample in enumerate(samples): 
                h.append(ROOT.TH1F('h%i' %i,'h%i' %i,nbin,minbin,maxbin))
                Plot(canv, Legend, dir, h[i], sample, '%s' %var, totsel, i, period = year, normalize=norm, stack = hs, ratio=doratio, sigGoodPV = args.sigGoodPV)
                if sample != 'ggHyyd' and sample != 'data': hbkg.Add(h[i])
                if sample == 'ggHyyd': hsig.Add(h[i])
                if sample == 'data': hdata.Add(h[i])
                if var == 'vertex': print (sample, h[i].GetBinContent(2)/h[i].Integral())
                if var == 'goodPV': print(sample,(h[i].GetBinContent(2)/h[i].Integral()))

            
            # --- Draw THStack if needed --- #
            
            if stack : 

                hbkg.SetLineColor(ROOT.kBlack)
                hbkg.SetMarkerSize(0)
                hbkg.SetFillStyle(3004)
                hbkg.SetFillColor(ROOT.kBlack)
                hsig.SetLineColor(ROOT.kRed)
                hsig.SetMarkerSize(0)
                print("total bkg: " , hbkg.Integral())
                
                if data:
                    hdata.SetLineColor(ROOT.kBlack)
                    hdata.SetMarkerSize(1)
                    hdata.SetMarkerColor(ROOT.kBlack)
                    hdata.SetLineColor(ROOT.kBlack)
                    print("data: " , hdata.Integral())

                hs.Draw('HIST')
                hs.GetXaxis().SetTitle(var_dict[var]['title'])
                hs.GetXaxis().SetLabelSize(0.05)
                hs.GetXaxis().SetTitleSize(0.05)
                hs.GetYaxis().SetTitleOffset(1.2)
                hs.GetYaxis().SetTitle('Events')
                hs.GetYaxis().SetTitleSize(0.05)
                hbkg.Draw('sameE2')
                hsig.Draw('sameHISTE')
                if blind: Blind(hdata,hsig)
                hdata.Draw('sameE2')
            


            ##### DRAW RATIO PLOT #####

            if doratio:
                ratio = []  # Vector or ratio histograms, one per sample
                canv.cd(2)
                # --- data/MC ---#
                if data:
                    line = ROOT.TLine(minbin,1,maxbin,1)
                    ratio.append(ROOT.TH1F('ratio','ratio',nbin,minbin,maxbin))
                    PlotRatio(canv, hdata, hbkg,ratio[0], '%s' %var, ROOT.kBlack,normalize=rnorm, significance=rsig, refline = line)
                # --- Significance ---#
                if (rsig and stack):  
                    ratio.append(ROOT.TH1F('ratio','ratio',nbin,minbin,maxbin))
                    PlotRatio(canv, hsig, hbkg,ratio[0], '%s' %var,  ROOT.kBlack,normalize=rnorm, significance=rsig)
                # --- ratio among each background and signal ---#
                else: 
                    if not data:
                        for i,sample in enumerate(samples): 
                            ratio.append(ROOT.TH1F('hratio%i' %i,'hratio%i' %i,nbin,minbin,maxbin))
                            if sample != 'ggHyyd': PlotRatio(canv, hsig, h[i],ratio[i], '%s' %var, samples_dict[sample]['color'], normalize=rnorm, significance=rsig)
                
            canv.cd(1)

            Legend.Draw()
            atlas_label.Draw()

            canv.SaveAs('%s_%s%s_%s.png' %(var,sel, 'Data' if data else '', tag))
        
