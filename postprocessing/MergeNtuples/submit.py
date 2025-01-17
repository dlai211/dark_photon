#! /usr/bin/env python

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



# cpp merge code
#'''user.piazza.v2.data23_13p6TeV.periodAllYear.physics_Main_minitrees.root#

def haddHisto(hist):
  os.chdir(dir)
  os.system('rucio get %s' %hist)
  os.system('hadd %s.root %s/*.root' %(hist,hist))
  os.chdir('..')

def getFiles(input):

  FilesStream = subprocess.run("rucio list-file-replicas %s" %input, capture_output=True, text=True,shell=True)
  FilesList = FilesStream.stdout.strip().split("\n")

  files = []
  #print(FilesList)
  for f in FilesList:
    if "INFN-MILANO_LOCALGROUPDISK" in f:
      files.append(f.split("INFN-MILANO_LOCALGROUPDISK: ")[1].replace(" ","").replace("|","")) #.replace("davs://storm-ft.mi.infn.it:8443/webdav","/gpfs/storage_4"))
  print(files)
  return files


inputs = [
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801174.Py8EG_A14NNPDF23LO_jj_JZ9incl_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801173.Py8EG_A14NNPDF23LO_jj_JZ8_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801172.Py8EG_A14NNPDF23LO_jj_JZ7_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801171.Py8EG_A14NNPDF23LO_jj_JZ6_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801170.Py8EG_A14NNPDF23LO_jj_JZ5_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801169.Py8EG_A14NNPDF23LO_jj_JZ4_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801166.Py8EG_A14NNPDF23LO_jj_JZ1_minitrees.root',
'user.sridouan.V01_truthJet1.mc23_13p6TeV.801165.Py8EG_A14NNPDF23LO_jj_JZ0_minitrees.root',
]
 
dir = sys.argv[1]
for input in inputs:
  histofile = input.replace('minitrees.root', 'hist')
  if not os.path.isfile('%s/%s.root' %(dir,histofile)): haddHisto(histofile)
  #haddHisto(histofile)

skim = sys.argv[2]

submit_file = open("submit_%s" %skim,"w")
submit_file.write("executable  = run_add_weight.sh \n")
submit_file.write("arguments = $(Item) $(file) $(skim)\n")
submit_file.write("log = submitdir/$(Item)_$(skim)_$(Process).log \n")
submit_file.write("output = submitdir/$(Item)_$(skim)_$(Process).out \n")
submit_file.write("error = submitdir/$(Item)_$(skim)_$(Process).err \n")
submit_file.write("initialdir = %s \n" %dir)
submit_file.write("should_transfer_files = YES \n")
submit_file.write("+JobFlavour = \"nextweek\"\n ")
submit_file.write("queue Item,file,skim from (\n")


for input in inputs:
  input_orig = input
  dataset = input.split('_minitrees')[0]
  dirname = '%s/%s_minitrees_%s.merged.root' %(dir,dataset,skim)
  #xos.system('rm -r %s' %dirname)
  #os.mkdir(dirname)
  input=input_orig
  print('%s_minitrees.root' %dataset)
  files = getFiles('%s_minitrees.root' %dataset)
  for f in files:
    print (f)
    submit_file.write("%s_minitrees.root,%s,%s\n" %(dataset,f,skim))
submit_file.write(")")
submit_file.close()
os.system('condor_submit submit_%s' %skim)
