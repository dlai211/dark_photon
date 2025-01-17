#! /usr/bin/env python

"""
Merge script for the monophoton minitrees (grid)
------------------------------------------------

Input: Text file with the list of paths of the _minitrees.root directories you want to merge (not need to be hadded)

The "_hist" files should be in the same directory with the same name but with the _hist extension instead of _minitrees.root

For example:
python normalize.py to_merge.txt xsec_file

where to_merge.txt contain the *_minitrees.root full path

"""

import os
import re
import sys
import glob
import ROOT
from ROOT import TLorentzVector 
from os import listdir
from os.path import isfile, join
import csv
from array import array
import subprocess
import math



# cpp merge code
#user.piazza:user.piazza.v2.data23_13p6TeV.periodAllYear.physics_Main_minitrees.root#

def MergeTreeHist(input_file, input_dataset, input_path_hist, xsec_file, output_file, old_tree_name,new_tree_name, is_nominal, skim):

  print("Merging tree " , old_tree_name)
  print("--- ")
  print("Input dataset : ", input_dataset)
  print("Input hist path : " , input_path_hist)
  print("Input tree name : " , old_tree_name)
  print("Input cross-section file : " , xsec_file)
  print("Output path     : " , output_file )
  print("Output tree name: " , new_tree_name)
  print("--- " )

  old_tree = ROOT.TChain(old_tree_name)
  old_tree.Add(input_file)
  old_tree.SetBranchStatus("*",  1)
  if 'data' not in input_dataset:old_tree.SetBranchStatus("pv_truth_x",0)
  if 'data' not in input_dataset:old_tree.SetBranchStatus("pv_truth_y",0)
  #old_tree.SetAutoFlush(0)
  #old_tree.SetAutoSave(0)

  if (old_tree.GetEntries() == 0):
    return

  old_tree.GetEntry(1)
  runNumber=old_tree.run
  print (runNumber)

  if 'data' not in input_dataset:
    weight_mc_sum_tmp= 0.
    xsec_ami_tmp= 0.
    filter_eff_ami_tmp= 0.
    kfactor_ami_tmp= 0.

    # Get sumw
    f_h = ROOT.TFile.Open(input_path_hist)
    h_sumw = f_h.Get("histoEventCount")
    weight_mc_sum_tmp = h_sumw.GetBinContent(1)
    f_h.Close()

    # Get xsec ecc...
    with open(xsec_file, mode='r') as csv_file:
      csv_reader = csv.DictReader(csv_file,delimiter = ' ')
      for row in csv_reader:
          #print (row['Run'])
          if int(row['Run']) == runNumber: 
            xsec_ami_tmp,filter_eff_ami_tmp,kfactor_ami_tmp=float(row['xsec']), float(row['filter']),float(row['kfactor'])
            #print (row)

  # Clone tree in new file
  print("Cloning ... " )
  print (output_file)

  #if 'data' not in input_tree:
  year= array('i',[0])
  if 'data' not in input_dataset:weight_mc_sum= array('f',[0])
  if 'data' not in input_dataset:xsec_ami= array('f',[0])
  if 'data' not in input_dataset:filter_eff_ami= array('f',[0])
  if 'data' not in input_dataset:kfactor_ami= array('f',[0])
  #corr_met_tst_et= array('f',[0])
  #corr_met_tst_phi= array('f',[0])
  #corr2_met_tst_et= array('f',[0])
  #corr2_met_tst_phi= array('f',[0])
  goodPV= array('i',[0])
  dmet= array('f',[0])
  dphi_jj= array('f',[0])
  m_jj= array('f',[0])
  dphi_central_jj= array('f',[0])
  m_central_jj= array('f',[0])
  dphi_met_phterm= array('f',[0])
  dphi_met_jetterm= array('f',[0])
  central_balance= array('f',[0])
  vector_central_balance= array('f',[0])
  balance= array('f',[0])
  vector_balance= array('f',[0])
  jet_central_pt= ROOT.std.vector('double')()
  jet_central_eta= ROOT.std.vector('double')()
  jet_central_phi= ROOT.std.vector('double')()
  jet_central_m= ROOT.std.vector('double')()
  jet_central_timing= ROOT.std.vector('double')()
  jet_central_emfrac= ROOT.std.vector('double')()
  jet_central_jvt= ROOT.std.vector('double')()
  jet_central_passJVT= ROOT.std.vector('double')()
  jet_central_isBjet= ROOT.std.vector('double')()
  if 'data' not in input_dataset:jet_central_PartonTruthLabelID= ROOT.std.vector('double')()
  jet_fwd_pt= ROOT.std.vector('double')()
  jet_fwd_eta= ROOT.std.vector('double')()
  jet_fwd_phi= ROOT.std.vector('double')()
  jet_fwd_m= ROOT.std.vector('double')()
  jet_fwd_timing= ROOT.std.vector('double')()
  jet_fwd_emfrac= ROOT.std.vector('double')()
  jet_fwd_jvt= ROOT.std.vector('double')()
  jet_fwd_passJVT= ROOT.std.vector('double')()
  jet_fwd_isBjet= ROOT.std.vector('double')()
  if 'data' not in input_dataset:jet_fwd_PartonTruthLabelID= ROOT.std.vector('double')()

  new_file = ROOT.TFile.Open(output_file, "recreate") if is_nominal else ROOT.TFile(output_file, "update")
  new_tree = old_tree.CloneTree(0)
  new_tree.SetName(new_tree_name)
  '''new_t.Write()
  new_file.Close()

  new_file = ROOT.TFile.Open(output, "update")
  new_tree = new_file.Get(new_tree_name)'''
  new_tree.SetBranchStatus("*",1)
  new_tree.SetBranchStatus("jet_*",0)


  branch_year = new_tree.Branch("year", year, "year/I")
  if 'data' not in input_dataset:branch_weight_mc_sum = new_tree.Branch("mc_weight_sum", weight_mc_sum, "mc_weight_sum/F")
  if 'data' not in input_dataset:branch_xsec = new_tree.Branch("xsec_ami", xsec_ami, "xsec_ami/F")
  if 'data' not in input_dataset:branch_filter = new_tree.Branch("filter_eff_ami", filter_eff_ami, "filter_eff_ami/F")
  if 'data' not in input_dataset:branch_kfactor = new_tree.Branch("kfactor_ami", kfactor_ami, "kfactor_ami/F")
  ##branch_corr_met_tst_et = new_tree.Branch("corr_met_tst_et", corr_met_tst_et, "corr_met_tst_et/F")
  ##branch_corr_met_tst_phi = new_tree.Branch("corr_met_tst_phi", corr_met_tst_phi, "corr_met_tst_phi/F")
  ##branch_corr2_met_tst_et = new_tree.Branch("corr2_met_tst_et", corr2_met_tst_et, "corr2_met_tst_et/F")
  ##branch_corr2_met_tst_phi = new_tree.Branch("corr2_met_tst_phi", corr2_met_tst_phi, "corr2_met_tst_phi/F")
  #branch_goodPV = new_tree.Branch("goodPV", goodPV, "goodPV/I")
  branch_dmet = new_tree.Branch("dmet", dmet , "dmet/F")
  branch_dphi_met_jetterm = new_tree.Branch("dphi_met_jetterm", dphi_met_jetterm , "dphi_met_jetterm/F")
  branch_dphi_met_phterm = new_tree.Branch("dphi_met_phterm", dphi_met_phterm , "dphi_met_phterm/F")
  branch_dphi_jj = new_tree.Branch("dphi_jj", dphi_jj, "dphi_jj/F")
  branch_m_jj = new_tree.Branch("m_jj", m_jj, "m_jj/F")
  branch_dphi_central_jj = new_tree.Branch("dphi_central_jj", dphi_central_jj, "dphi_central_jj/F")
  branch_m_central_jj = new_tree.Branch("m_central_jj", m_central_jj, "m_central_jj/F")
  branch_central_balance = new_tree.Branch("central_balance", central_balance, "central_balance/F")
  branch_vector_central_balance = new_tree.Branch("vector_central_balance", vector_central_balance, "vector_central_balance/F")
  branch_vector_balance = new_tree.Branch("vector_balance", vector_balance, "vector_balance/F")
  branch_balance = new_tree.Branch("balance", balance, "balance/F")
  #branch_n_jet_central = new_tree.Branch("n_jet_central", n_jet_central, "n_jet_central/I")
  #branch_jet_central_vecsumpt = new_tree.Branch("jet_central_vecsumpt", jet_central_vecsumpt, "jet_central_vecsumpt/F")
  branch_jet_central_pt= new_tree.Branch("jet_central_pt", jet_central_pt)
  branch_jet_central_eta= new_tree.Branch("jet_central_eta", jet_central_eta)
  branch_jet_central_phi= new_tree.Branch("jet_central_phi", jet_central_phi)
  branch_jet_central_m= new_tree.Branch("jet_central_m", jet_central_m)
  branch_jet_central_timing= new_tree.Branch("jet_central_timing", jet_central_timing)
  branch_jet_central_emfrac= new_tree.Branch("jet_central_emfrac", jet_central_emfrac)
  branch_jet_central_jvt= new_tree.Branch("jet_central_jvt", jet_central_jvt)
  branch_jet_central_passJVT= new_tree.Branch("jet_central_passJVT", jet_central_passJVT)
  branch_jet_central_isBjet= new_tree.Branch("jet_central_isBjet", jet_central_isBjet)
  if 'data' not in input_dataset:branch_jet_central_PartonTruthLabelID= new_tree.Branch("jet_central_PartonTruthLabelID", jet_central_PartonTruthLabelID)
  branch_jet_central_pt= new_tree.Branch("jet_fwd_pt", jet_fwd_pt)
  branch_jet_fwd_eta= new_tree.Branch("jet_fwd_eta", jet_fwd_eta)
  branch_jet_fwd_phi= new_tree.Branch("jet_fwd_phi", jet_fwd_phi)
  branch_jet_fwd_m= new_tree.Branch("jet_fwd_m", jet_fwd_m)
  branch_jet_fwd_timing= new_tree.Branch("jet_fwd_timing", jet_fwd_timing)
  branch_jet_fwd_emfrac= new_tree.Branch("jet_fwd_emfrac", jet_fwd_emfrac)
  branch_jet_fwd_jvt= new_tree.Branch("jet_fwd_jvt", jet_fwd_jvt)
  branch_jet_fwd_passJVT= new_tree.Branch("jet_fwd_passJVT", jet_fwd_passJVT)
  branch_jet_fwd_isBjet= new_tree.Branch("jet_fwd_isBjet", jet_fwd_isBjet)
  if 'data' not in input_dataset:branch_jet_fwd_PartonTruthLabelID= new_tree.Branch("jet_fwd_PartonTruthLabelID", jet_fwd_PartonTruthLabelID)


  year[0]=2016
  if 'mc20d' in input_dataset: year[0]=2018
  if 'mc20e' in input_dataset: year[0]=2017
  if 'mc21' in input_dataset: year[0]=2022
  if 'mc23' in input_dataset: year[0]=2023
  nEntries = old_tree.GetEntries()
  #Filling tree
  print("Filling tree... " )
  for iEnt in range(nEntries):
    old_tree.GetEntry(iEnt)
    if (iEnt%1000 == 0): print(iEnt)

    jet_central_pt.clear()
    jet_central_eta.clear()
    jet_central_phi.clear()
    jet_central_m.clear()
    jet_central_timing.clear()
    jet_central_emfrac.clear()
    jet_central_jvt.clear()
    jet_central_passJVT.clear()
    jet_central_isBjet.clear()
    if 'data' not in input_dataset:jet_central_PartonTruthLabelID.clear()
    jet_fwd_pt.clear()
    jet_fwd_eta.clear()
    jet_fwd_phi.clear()
    jet_fwd_m.clear()
    jet_fwd_timing.clear()
    jet_fwd_emfrac.clear()
    jet_fwd_jvt.clear()
    jet_fwd_passJVT.clear()
    jet_fwd_isBjet.clear()
    if 'data' not in input_dataset:jet_fwd_PartonTruthLabelID.clear()

    #pass_photon = ((old_tree.n_ph_loose == 1 and old_tree.n_ph_baseline == 0) or (old_tree.n_ph_baseline >= 1))
    pass_photon = (old_tree.n_ph_baseline >= 1)
    passSkim = False

    if skim == 'ey' : passSkim = old_tree.n_el==1 and old_tree.n_mu_baseline==0 and pass_photon
    if skim == 'eey' : passSkim = old_tree.n_el==2 and old_tree.n_mu_baseline==0 and pass_photon
    if skim == 'uy' : passSkim = old_tree.n_el_baseline==0 and old_tree.n_mu==1 and pass_photon
    if skim == 'uuy' : passSkim = old_tree.n_el_baseline==0 and old_tree.n_mu==2 and pass_photon
    if skim == 'y' : passSkim = old_tree.n_el_baseline==0 and old_tree.n_mu_baseline==0 and pass_photon
    if skim == 'efake':
      pass_ee = old_tree.n_el==2 and old_tree.n_mu_baseline==0 and old_tree.n_ph_baseline == 0
      pass_eee = old_tree.n_el==3 and old_tree.n_mu_baseline==0 and old_tree.n_ph_baseline == 0
      pass_ue = old_tree.n_el==1 and old_tree.n_mu==1 and old_tree.n_ph_baseline == 0
      pass_uue = old_tree.n_el==1 and old_tree.n_mu==2 and old_tree.n_ph_baseline == 0
      pass_e = old_tree.n_el==1 and old_tree.n_mu_baseline==0 and old_tree.n_ph_baseline == 0
      passSkim = pass_ee or pass_eee or pass_ue or pass_uue or pass_e 


    if (passSkim):

      #if 'data' not in input_tree:
      if 'data' not in input_dataset:weight_mc_sum[0]= weight_mc_sum_tmp
      if 'data' not in input_dataset:xsec_ami[0]= xsec_ami_tmp
      if 'data' not in input_dataset:filter_eff_ami[0]= filter_eff_ami_tmp
      if 'data' not in input_dataset:kfactor_ami[0]= kfactor_ami_tmp
      #corr_met_tst_et[0] = CorrMet(old_tree)[0]
      #corr_met_tst_phi[0] = CorrMet(old_tree)[1]
      #corr2_met_tst_et[0] = CorrMet2(old_tree)[0]
      #corr2_met_tst_phi[0] = CorrMet2(old_tree)[1]
      if 'data' not in input_dataset:goodPV[0]= PV(old_tree)
      dmet[0] = Dmet(old_tree)
      dphi_met_jetterm[0] = DphiMetJetterm(old_tree)
      dphi_met_phterm[0] = DphiMetPhterm(old_tree)
      dphi_jj[0] = jj(old_tree)[0]
      m_jj[0] = jj(old_tree)[1]
      dphi_central_jj[0] = jj(old_tree, central=True)[0]
      m_central_jj[0] = jj(old_tree, central=True)[1]
      central_balance[0] = CentralBalance(old_tree)[0]
      vector_central_balance[0] = VectorCentralBalance(old_tree)[0]
      vector_balance[0] = VectorBalance(old_tree)
      #jet_central_vecsumpt[0] = VectorCentralBalance(old_tree)[1]
      balance[0] = Balance(old_tree)[0]
      #n_jet_central[0] = SplitJet(old_tree, jet_central_pt, jet_fwd_pt, 'pt')
      SplitJet(old_tree, jet_central_pt, jet_fwd_pt, 'pt')
      SplitJet(old_tree, jet_central_eta, jet_fwd_eta, 'eta')
      SplitJet(old_tree, jet_central_phi, jet_fwd_phi, 'phi')
      SplitJet(old_tree, jet_central_m, jet_fwd_m, 'm')
      SplitJet(old_tree, jet_central_timing, jet_fwd_timing, 'timing')
      SplitJet(old_tree, jet_central_emfrac, jet_fwd_emfrac, 'emfrac')
      SplitJet(old_tree, jet_central_jvt, jet_fwd_jvt, 'jvt')
      SplitJet(old_tree, jet_central_passJVT, jet_fwd_passJVT, 'passJVT')
      SplitJet(old_tree, jet_central_isBjet, jet_fwd_isBjet, 'isBjet')
      #if 'data' not in input_dataset:SplitJet(old_tree, jet_central_PartonTruthLabelID, jet_fwd_PartonTruthLabelID, 'PartonTruthLabelID')

      #print('new jet central pt', new_tree.jet_central_pt)
      #if (passSkim): new_tree.Fill()
      new_tree.Fill()

  print("end of fill trees ... " )
  
  #new_file = ROOT.TFile.Open(output, "update") 

  #new_file.Delete("nominal;*")
  new_tree.Write(new_tree_name, ROOT.TObject.kOverwrite)
  new_file.Close()
  #new_file_ey = ROOT.TFile.Open(output_ey, "update")



def PV(t):
  return (abs(t.pv_truth_z.at(0)-t.pv_z.at(0)) <= 0.3)

def jj(t, central=False):
  dphi = -10
  m=-10
  jet1=TLorentzVector(0,0,0,0)
  jet2=TLorentzVector(0,0,0,0)
  if t.jet_pt.size() >=2: 
    index = []
    for i in range(t.jet_pt.size()):
      if len(index) < 2:
        if abs(t.jet_eta[i])<=2.5 or not central: 
            index.append(i)
    if len(index)==2:
      jet1.SetPtEtaPhiM(t.jet_pt.at(index[0]), t.jet_eta.at(index[0]),t.jet_phi.at(index[0]),t.jet_m.at(index[0]))
      jet2.SetPtEtaPhiM(t.jet_pt.at(index[1]), t.jet_eta.at(index[1]),t.jet_phi.at(index[1]),t.jet_m.at(index[1]))
      #diffjet = TLorentzVector(jet1-jet2)
      sumjet = TLorentzVector(jet1+jet2)
      dphi = jet1.DeltaPhi(jet2)
      m = sumjet.M()
  return dphi, m

def Dmet(t):
  return t.met_tst_noJVT_et-t.met_tst_et


def DphiMetJetterm(t):
  dphi = -10
  if t.met_jetterm_et > 0: 
    met = TLorentzVector(0,0,0,0)
    jetterm = TLorentzVector(0,0,0,0)
    met.SetPtEtaPhiM(t.met_tst_et, 0,t.met_tst_phi,0)
    jetterm.SetPtEtaPhiM(t.met_jetterm_et, 0,t.met_jetterm_phi,0)
    dphi = met.DeltaPhi(jetterm)
  return dphi

def DphiMetPhterm(t):
  dphi = -10
  if t.met_phterm_et > 0: 
    met = TLorentzVector(0,0,0,0)
    phterm = TLorentzVector(0,0,0,0)
    met.SetPtEtaPhiM(t.met_tst_et, 0,t.met_tst_phi,0)
    phterm.SetPtEtaPhiM(t.met_phterm_et, 0,t.met_phterm_phi,0)
    dphi = met.DeltaPhi(phterm)
  return dphi


def CentralBalance(t):
  balance = -10
  ph_pt = 0
  sumptjet = 0
  N = 0
  if t.ph_pt.size()>0: 
    ph_pt = t.ph_pt.at(0)
    met = t.met_tst_et
    sumptjet = 0
    N=0
    for i,jp in enumerate(t.jet_pt):
      if abs(t.jet_eta.at(i))<2.8: 
        sumptjet+=jp
        N+=1
    if sumptjet > 0: balance = (ph_pt+met)/sumptjet
    
  return balance, sumptjet, N


def VectorCentralBalance(t):
  balance = -10
  sumptjet = -10
  if t.met_phterm_et > 0 :
    met = TLorentzVector(t.met_tst_et, 0,t.met_tst_phi,0)
    mety = TLorentzVector(t.met_phterm_et, 0,t.met_phterm_phi,0)
    jet = TLorentzVector(0,0,0,0)

    MetMety = TLorentzVector(met+mety)
    
    for i,jp in enumerate(t.jet_pt):
      if abs(t.jet_eta.at(i))<=2.8: 
        jet_tmp = TLorentzVector(t.jet_pt.at(i), t.jet_eta.at(i),t.jet_phi.at(i),t.jet_m.at(i))
        jet = TLorentzVector(jet+jet_tmp)
    sumptjet = jet.Pt() 
    if sumptjet > 0: balance = (MetMety.Pt())/sumptjet
      
  return balance, sumptjet


def VectorBalance(t):
  balance = -10
  sumptjet = -10
  if t.met_phterm_et > 0 :
    met = TLorentzVector(t.met_tst_et, 0,t.met_tst_phi,0)
    mety = TLorentzVector(t.met_phterm_et, 0,t.met_phterm_phi,0)
    jet = TLorentzVector(0,0,0,0)

    MetMety = TLorentzVector(met+mety)
    
    for i,jp in enumerate(t.jet_pt):
      jet_tmp = TLorentzVector(t.jet_pt.at(i), t.jet_eta.at(i),t.jet_phi.at(i),t.jet_m.at(i))
      jet = TLorentzVector(jet+jet_tmp)
    sumptjet = jet.Pt() 
    if sumptjet > 0: balance = (MetMety.Pt())/sumptjet
      
  return balance


def Balance(t):
  ph_pt = 0
  balance = -10
  sumptjet = 0
  if t.ph_pt.size()>0: 
    ph_pt = t.ph_pt.at(0)
    met = t.met_tst_et
    for i,jp in enumerate(t.jet_pt):
      sumptjet+=jp
    if sumptjet>0: balance = (ph_pt+met)/sumptjet
      
  return balance, sumptjet

def SplitJet(t, jet_central, jet_fwd, var):
  var_dict = {}
  var_dict['pt'] = t.jet_pt
  var_dict['phi'] = t.jet_phi
  var_dict['eta'] = t.jet_eta
  var_dict['m'] = t.jet_m
  var_dict['timing'] = t.jet_timing
  var_dict['emfrac'] = t.jet_emfrac
  var_dict['jvt'] = t.jet_jvt
  var_dict['passJVT'] = t.jet_passJVT
  var_dict['isBjet'] = t.jet_isBjet
  #if t.jet_PartonTruthLabelID:var_dict['PartonTruthLabelID'] = t.jet_PartonTruthLabelID
  N=0
  for i in range(t.jet_pt.size()):
    if abs(t.jet_eta[i])<=2.8: 
      jet_central.push_back(var_dict[var][i])
      N=N+1
    else: jet_fwd.push_back(var_dict[var][i])
  return N


input_minitrees = sys.argv[1]
xsec_file = sys.argv[2]
files_loc = sys.argv[3]
skim = sys.argv[4]
input_file = sys.argv[5]
in_output_path = sys.argv[6]
sample = 'nominal'
f = ROOT.TFile.Open(input_file)#
input_trees = [ b.GetName() for b in f.GetListOfKeys() ]
#print ('filename', f.GetName())
f.Close()


for tname in input_trees:

  if 'ggH_H125_yyv_myv' in input_minitrees: sample='ggHyyd'
  if 'Sh_2211_Z' in input_minitrees or 'Sh_2214_Z' in input_minitrees: sample='Zjets'
  if 'Sh_2211_W' in input_minitrees or 'Sh_2214_W' in input_minitrees: sample='Wjets'
  if 'munugamma' in input_minitrees or 'enugamma' in input_minitrees : sample='Wgamma'
  if 'mumugamma' in input_minitrees or 'eegamma' in input_minitrees or 'nunugamma' in input_minitrees: sample='Zgamma'
  if 'SinglePhoton' in input_minitrees or 'gammajet' in input_minitrees: sample='gammajets'
  if 'JZ' in input_minitrees: sample='dijets'
  if 'data' in input_minitrees: sample='data'
  old_tree_name = tname
  new_tree_name = tname
  #if 'nominal' in tname:
  #  new_tree_name=tname.replace('nominal',sample)
  #else:
  #  new_tree_name = tname

input_minitrees_orig = input_minitrees
input_file_orig = input_file
output_path_partial = input_minitrees.replace('.root','_%s.merged.root' %skim)
output_path = '%s/%s' %(in_output_path, output_path_partial)
#os.system('rm -r %s' %(output_path))
if not os.path.isdir(output_path) : os.mkdir(output_path)
output_file = '%s/%s' % (output_path,input_file.split('._')[1])
input_minitrees = input_minitrees_orig
input_file = input_file_orig

is_nominal = not ('syst' in tname)
print (input_minitrees, input_file)
print (output_file)
input_hists_partial = input_minitrees.replace('minitrees.root','hist.root')
input_hists = '%s/%s' %(in_output_path, input_hists_partial)
input_minitrees = input_minitrees_orig
MergeTreeHist(input_file, input_minitrees, input_hists, xsec_file, output_file, old_tree_name, new_tree_name, is_nominal, skim)
