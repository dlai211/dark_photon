import os,sys
import ROOT
import math
import re

sys.path.append('PlotUtilities.py')
import PlotUtilities
from PlotUtilities import *


sys.path.append('Dictionaries.py')
import Dictionaries
from Dictionaries import *

import argparse

ROOT.gROOT.SetStyle("ATLAS")
ROOT.gStyle.SetOptStat(0)



def ExtractNormFactors(hdata, hmc):
    hnorm = hdata.Clone()
    hnorm.Divide(hmc)
    k = {}
    for b in range(hnorm.GetNbinsX()):
        k[b] = {'min': hnorm.GetBinLowEdge(b), 'max': hnorm.GetBinLowEdge(b)+hnorm.GetBinWidth(b), 'k': hnorm.GetBinContent(b), 'err': hnorm.GetBinError(b)}
    return k[b]


def VBackground(samples, region, var, file_path, period):
    samples_dict = getSampleDict()

    varstr = var_dict[var]['var']
    nbin,minbin,maxbin = GetBinning(var)
    allsamples = ['gammajets_direct','gammajets_frag','dijets','Wjets','Zjets','Wgamma','Zgamma','data']
    hbkg = ROOT.TH1F('hbkg','hbkg',nbin,minbin,maxbin)
    hsubtract = ROOT.TH1F('hsub','hsub',nbin,minbin,maxbin)
    hdata = ROOT.TH1F('hdata','hdata',nbin,minbin,maxbin)
    h = []
    for i,s in enumerate(allsamples):

        chain = ROOT.TChain(samples_dict[s]['tree'])
        fillChain(chain, file_path, period, s, sel = 'uuy')
        weight = getWeight(period, s)
        if s == 'ggHyyd' and sigGoodPV: weight = weight+'*goodPV' #to show only signal with good PV
        sel = GetSelString(region)
        h.append(ROOT.TH1F('h%i' %i,'h%i' %i,nbin,minbin,maxbin))
        chain.Draw('(%s) >> %s' %(varstr,h[i].GetName()), '(%s) * (%s)' %(sel,weight))
        if s in samples: hbkg.Add(h[i])
        elif s!= 'data': hsubtract.Add(h[i])
        elif s == 'data': hdata.Add(h[i])
    hdata.Add(hsubtract,-1)
    k_dict = ExtractNormFactors(hdata, hbkg)

    return k_dict

def jetFakeBackground():
    ff_dict = getReweightDict()
    reweight = getReweighting(ff_dict, 'abs(ph_eta[0])')
    return reweight

def getReweighting(dict, var):
    reweights = []
    for bin in dict:
        reweights.append('(%s>%s && %s<%s)*%s' %(var, dict[bin]['min'], var, dict[bin]['max'], dict[bin]['w']))
    rewstring = " + ".join(reweights)
    return '(%s)' %rewstring