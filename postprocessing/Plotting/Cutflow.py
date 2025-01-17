
import sys
import PlotUtilities
from PlotUtilities import *
import os
import ROOT


parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--samples', type=str, nargs='+', help='list of samples', default=['all'])
parser.add_argument('-c', '--cuts', type=str, nargs='+', help='list of selections', default=['MET100_phPT50'])
parser.add_argument('-p', '--period', type=str, help='period (Run2,mc23d)', default='mc23d')
parser.add_argument('-d', '--dir', type=str, help='input directory', default='/data/fpiazza/ggHyyd/NtuplesSelaiman/')



if __name__ == '__main__':

    args=parser.parse_args()
    period = args.period
    dir = args.dir
    sel = args.cuts
    samples = args.samples if ars.samples!=['all'] else ['Zgamma','Wgamma','Zjets_e','Zjets_j','Wjets_e','Wjets_j','gammajets_direct','gammajets_frag','dijets']
    lumi = getLumi(period)

    Cutflow(input_dir, samples, sel, period)
