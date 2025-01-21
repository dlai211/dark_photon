import os
import math
import ROOT

def getVarDict():
    dict = {}
    dict['vtx_sumPt']=              {'var':'vtx_sumPt','bins':[50,0,100000], 'title': 'vtx_sumPt', 'shift':'+0'}
    # dict['puWeight']=               {'var':'pu_weight','bins':[50,0,2], 'title': 'PU weight', 'shift':'+0'}#150*(1-1/0.7)'}
    # dict['actualIntPerXing']=       {'var':'actualIntPerXing','bins':[50,0,100], 'title': '<#mu>', 'shift':'+0'}#150*(1-1/0.7)'}
    # dict['mt']=                     {'var':'(mt/1000)','bins':[15,0,300], 'title': 'm_{T} [GeV]', 'shift':'+0'}#150*(1-1/0.7)'}
    # dict['metsig']=                 {'var':'met_tst_sig','bins':[15,0,30], 'title': 'E_{T}^{miss} significance', 'shift':'*1'}#((met_tst_et+50000)/met_tst_et)'}
    # dict['metsigres']=              {'var':'met_tst_et/met_tst_sig','bins':[50,0,100000], 'title': 'E_{T}^{miss} significance', 'shift':'*1'}
    # dict['met']=                    {'var':'met_tst_et','bins':[50,0,300000], 'title': 'E_{T}^{miss} [GeV]','shift':'+50000'}
    # dict['met_noJVT']=              {'var':'met_tst_noJVT_et','bins':[50,0,300000], 'title': 'E_{T}^{miss} [GeV]'}
    # dict['met_cst']=                {'var':'met_cst_et','bins':[50,0,300000], 'title': 'E_{T}^{miss} CST [GeV]'}
    # dict['met_track']=              {'var':'met_track_et','bins':[50,0,300000], 'title': 'E_{T}^{miss} Track [GeV]'}
    # dict['dmet']=                   {'var':'(met_tst_noJVT_et-met_tst_et)','bins':[20,-100000,100000], 'title': 'E_{T,noJVT}^{miss}-E_{T}^{miss} [GeV]', 'shift':'*1'}  
    # dict['ph_pt']=                  {'var':'ph_pt[0]','bins':[50,0,300000],'title': 'p_{T}^{#gamma} [GeV]', 'shift':'-150000'}
    # dict['ph_eta']=                 {'var':'abs(ph_eta)[0]','bins':[16,0,4],'title': '#eta^{#gamma}'}
    # dict['ph_phi']=                 {'var':'ph_phi[0]','bins':[50,-4,4],'title': '#eta^{#gamma}'}
    # dict['jet_central_eta']=        {'var':'jet_central_eta[0]','bins':[50,-4,4],'title': '#eta^{#jets}'}
    # dict['jet_central_pt1']=        {'var':'jet_central_pt[0]','bins':[50,0,300000], 'title': 'p_{T}^{j1} [GeV]'}
    # dict['jet_central_pt2']=        {'var':'jet_central_pt[1]','bins':[50,0,300000], 'title': 'p_{T}^{j2} [GeV]'}
    # dict['jet_central_pt']=         {'var':'jet_central_pt','bins':[50,0,300000], 'title': 'p_{T}^{j} [GeV]'}
    # dict['dphi_met_phterm']=        {'var':'acos(cos(met_tst_phi-met_phterm_phi))','bins':[16,0,4], 'title': '#Delta#phi(E_{T}^{miss},E_{T}^{#gamma})','shift':'+0'}
    # dict['dphi_met_ph']=            {'var':'acos(cos(met_tst_phi-ph_phi[0]))','bins':[50,0,4], 'title': '#Delta#phi(E_{T}^{miss},E_{T}^{#gamma})'}
    # dict['dphi_met_jetterm']=       {'var':'((met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0)','bins':[16,0,4], 'title': '#Delta#phi(E_{T}^{miss},E_{T}^{#jet})'}
    # dict['dphi_phterm_jetterm']=    {'var':'((met_jetterm_et>0)*(acos(cos(met_phterm_phi-met_jetterm_phi))) + (met_jetterm_et==0)*4)','bins':[50,0,4], 'title': '#Delta#phi(E_{T}^{#gamma},E_{T}^{jet})'}
    # dict['dphi_ph_centraljet1']=    {'var':'acos(cos(ph_phi[0]-jet_central_phi[0]))','bins':[50,0,4], 'title': '#Delta#phi(#gamma,j1)'}
    # dict['dphi_ph_jet1']=           {'var':'acos(cos(ph_phi[0]-jet_central_phi[0]))','bins':[50,0,4], 'title': '#Delta#phi(#gamma,j1)'}
    # dict['dphi_central_jet1_jet2']= {'var':'acos(cos(jet_central_phi[0]-jet_central_phi[1]))','bins':[50,0,4], 'title': '#Delta#phi(j1,j2)'}
    # dict['metplusph']=              {'var':'(met_tst_et+ph_pt[0])','bins':[50,0,300000], 'title':'E_{T}^{miss}+p_{T}^{#gamma} [GeV]'}
    # dict['failJVT_jet_pt']=         {'var':'failJVT_jet_pt','bins':[50,0,300000], 'title':'p_{T}^{noJVT jet} [GeV]'}
    # dict['failJVT_jet_pt1']=        {'var':'failJVT_jet_pt[0]','bins':[40,20000,60000], 'title':'p_{T}^{noJVT jet1} [GeV]'}
    # dict['jet_jvt']=                {'var':'jet_jvt','bins':[50,0,300000], 'title':'Jet JVT'}
    # dict['softerm']=                {'var':'met_softerm_tst_et','bins':[50,0,100000], 'title': 'E_{T}^{soft} [GeV]'}
    # dict['jetterm']=                {'var':'met_jetterm_et','bins':[50,0,300000], 'title': 'E_{T}^{jet} [GeV]'}
    # dict['jetterm_sumet']=          {'var':'met_jetterm_sumet','bins':[50,0,300000], 'title': 'E_{T}^{jet} [GeV]'}
    # dict['n_jet']=                  {'var':'n_jet','bins':[10,0,10], 'title': 'N_{jet}'}
    # dict['n_jet_central']=          {'var':'n_jet_central','bins':[10,0,10], 'title': 'N_{jet}'}
    # dict['n_jet_fwd']=              {'var':'n_jet-n_jet_central','bins':[10,0,10], 'title': 'N_{jet}'}
    # dict['vertex']=                 {'var':'(abs(pv_truth_z[0]-pv_z[0]) == Min$(abs(pv_truth_z[0]-pv_z)))','bins':[2,0,2], 'title': 'good PV'}
    # dict['goodPV']=                 {'var':'(abs(pv_truth_z[0]-pv_z[0]) <= 0.5)','bins':[2,0,2], 'title': 'good PV'}
    # dict['dphi_met_central_jet']=   {'var':'acos(cos(met_tst_phi-jet_central_phi[0]))','bins':[50,0,4], 'title': '#Delta#phi(E_{T}^{miss},jet)'}
    # dict['counts']=                 {'var':'0.5','bins':[1,0,1], 'title': ''}
    # dict['jet_central_timing1']=    {'var':'jet_central_timing[0]','bins':[50,-40,40], 'title': 'Jet timing'}
    # dict['jet_central_timing']=     {'var':'jet_central_timing','bins':[50,-40,40], 'title': 'Jet timing'}
    # dict['jet_central_emfrac']=     {'var':'jet_central_emfrac','bins':[50,-1,2], 'title': 'Jet EM fraction'}
    # dict['jet_central_emfrac1']=    {'var':'jet_central_emfrac[0]','bins':[50,-1,2], 'title': 'Jet EM fraction'}
    # dict['balance']=                {'var':'(met_tst_et+ph_pt[0])/Sum$(jet_central_pt)','bins':[100,0,20], 'title': 'balance'}
    # dict['balance_sumet']=          {'var':'(met_tst_et+ph_pt[0])/met_jetterm_sumet','bins':[80,0,80], 'title': 'balance'}
    # dict['central_jets_fraction']=  {'var':'(n_jet>0)*(n_jet_central/n_jet)+(n_jet==0)*(-1)','bins':[50,-1,2], 'title': 'Central jets fraction'}#Alt$(Sum$(jet_central_pt)/met_jetterm_sumet,1.5)
    # dict['trigger'] =               {'var':'trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M','bins':[2,0,2], 'title': 'Pass Trigger'}
    # dict['dphi_jj']=                {'var':'Alt$(acos(cos(jet_central_phi[1]-jet_central_phi[0])),-1)','bins':[20,-1, 4], 'title': '#Delta#phi(j1,j2)'}

    return dict

def getSampleDict():
    dict = {}
    dict['Zjets'] =             {'color': ROOT.kGreen-2,    'legend': 'Z(#nu#nu,ll)+jets',                  'tree': 'nominal',   'filenames': ['Zjets']}
    dict['Zjets_j'] =           {'color': ROOT.kGreen-2,    'legend': 'Z(#nu#nu,ll)+jets (jfake)',          'tree': 'nominal',   'filenames': ['Zjets']}
    dict['Zjets_e'] =           {'color': ROOT.kPink,	    'legend': 'Z(#nu#nu,ll)+jets (efake)',          'tree': 'nominal',   'filenames': ['Zjets']}
    dict['Zgamma'] =            {'color': ROOT.kOrange+7,   'legend': 'Z(#nu#nu)+#gamma',                   'tree': 'nominal',   'filenames': ['Zgamma']}
    dict['Wgamma'] =            {'color': ROOT.kOrange+1,   'legend': 'W(l#nu)+#gamma',                     'tree': 'nominal',   'filenames': ['Wgamma']}
    dict['Wjets'] =             {'color': ROOT.kTeal+5,     'legend': 'W(l#nu)+jets',                       'tree': 'nominal',   'filenames': ['Wjets']}
    dict['Wjets_j'] =           {'color': ROOT.kTeal+5,     'legend': 'W(l#nu)+jets (jfake)',               'tree': 'nominal',   'filenames': ['Wjets']}
    dict['Wjets_e'] =           {'color': ROOT.kMagenta,    'legend': 'W(l#nu)+jets (efake)',               'tree': 'nominal',   'filenames': ['Wjets']}
    dict['gammajet_direct'] =   {'color': ROOT.kBlue+2,     'legend': '#gamma+jets direct',                 'tree': 'gammajets',   'filenames': ['gammajet_direct']}
    dict['gammajet_frag'] =     {'color': ROOT.kBlue-5,     'legend': '#gamma+jets frag',                   'tree': 'gammajets',   'filenames': ['gammajet_frag']}
    dict['dijet'] =             {'color': ROOT.kCyan+1,     'legend': 'multijets',                          'tree': 'dijets',   'filenames': ['dijet']}
    dict['jj'] =             {'color': ROOT.kCyan+1,     'legend': 'multijets',                          'tree': 'dijets',   'filenames': ['jj']}
    dict['ggHyyd'] =            {'color': ROOT.kRed,        'legend': 'ggH, H#rightarrow#gamma#gamma_{d}',  'tree': 'nominal',   'filenames': ['ggHyyd']}
    dict['VBF'] =               {'color': ROOT.kOrange,     'legend': 'VBF, H#rightarrow#gamma#gamma_{d}',  'tree': 'nominal',   'filenames': ['VBF']}
    dict['ZH'] =                {'color': ROOT.kGreen-2,    'legend': 'ZH, H#rightarrow#gamma#gamma_{d}',   'tree': 'nominal',   'filenames': ['ZH']}
    dict['WH'] =                {'color': ROOT.kBlue,       'legend': 'WH, H#rightarrow#gamma#gamma_{d}',   'tree': 'nominal',   'filenames': ['WH']}
    dict['fakeMET'] =           {'color': ROOT.kBlue,       'legend': 'Fake MET',                           'tree': 'nominal',   'filenames': ['data']}
    dict['data'] =              {'color': ROOT.kBlack,      'legend': 'Data',                               'tree': 'nominal',   'filenames': ['data']}
    dict['jetFake'] =           {'color': ROOT.kCyan+1,     'legend': 'jet#rightarrow#gamma',               'tree': 'nominal',   'filenames': ['data']}
    dict['fakeMETMC'] =         {'color': ROOT.kBlue,       'legend': 'Fake MET MC',                        'tree': 'nominal',   'filenames': ['gammajet_direct','gammajets_frag','dijets']}
    dict['yj'] =                {'color': ROOT.kBlue,       'legend': 'y+jets',                             'tree': 'nominal',   'filenames': ['gammajet_direct','gammajet_frag']}
    dict['jetFakeMC'] =         {'color': ROOT.kCyan+1,     'legend': 'jet#rightarrow#gamma',               'tree': 'nominal',   'filenames': ['gammajet_frag','dijet']}
    dict['subtract'] =          {'color': ROOT.kBlack,      'legend': 'Subtract bkg',                       'tree': 'nominal',   'filenames': ['Zgamma','Wgamma','Zjets','Wjets']}
    dict['subtractY'] =         {'color': ROOT.kBlack,      'legend': 'Subtract bkg',                       'tree': 'nominal',   'filenames':  ['Zgamma','Wgamma', 'Zjets_e','Wjets_e']}

    return dict

def getReweightDict( bins = [], weights = [],type = 'ff'):
    dict = {}
    if type == 'ff':
        dict = {}
        dict['bin1']={'min': 0, 'max': 0.6, 'w': 1.51}
        dict['bin2']={'min': 0.6, 'max': 1.37, 'w': 2.03}
        dict['bin3']={'min': 1.52, 'max': 1.81, 'w': 1.95}
        dict['bin4']={'min': 1.81, 'max': 2.37, 'w': 1.70}
    else:
        for b, bin in enumerate(range(len(bins)-1)):
            dict['bin%i' %b] = {'min': bins[b], 'max': bins[b+1], 'w': weights[b]}
    return dict
    
def getSelDict():

    iso =  '((ph_topoetcone40[0]-2450 ) / ph_pt[0]) < 0.022'
    noniso =  '((ph_topoetcone40[0]-2450 ) / ph_pt[0]) > 0.1'

    dict= {}
    
    dict['yIso'] = {'str' : 'n_ph_baseline == 1 && n_mu_baseline == 0 && n_el_baseline == 0 && n_jet<=4 && %s' %iso}
    dict['uuyIso'] = {'str' : 'n_ph_baseline == 1  && n_mu == 2 && n_el_baseline == 0 && n_jet<=4 && %s' %iso}
    dict['uyIso'] = {'str' : 'n_ph_baseline == 1  && n_mu == 1 && n_el_baseline == 0 && n_jet<=4 && %s' %iso}
    dict['yNonIso'] = {'str' : 'n_ph_baseline == 1 && n_mu_baseline == 0 && n_el_baseline == 0 && n_jet<=4 && %s' %noniso}
    dict['uuyNonIso'] = {'str' : 'n_ph_baseline == 1  && n_mu == 2 && n_el_baseline == 0 && n_jet<=4 && %s' %noniso}
    dict['uyNonIso'] = {'str' : 'n_ph_baseline == 1  && n_mu == 1 && n_el_baseline == 0 && n_jet<=4 && %s' %noniso}

    dict['met100phPT50'] = {'str' : 'met_tst_et > 100000 && trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M && ph_pt[0]>50000'}
    dict['met50phPT200'] = {'str' : 'met_tst_et > 50000 && trigger_HLT_g140_loose_L1eEM26M && ph_pt[0]>200000'}
    dict['mt80'] = {'str':'(mt/1000) >80'}
    dict['mt110'] = {'str':'mt >110000'}
    dict['mt140'] = {'str':'mt <140000'}
    dict['mt150'] = {'str':'mt <150000'}
    dict['mt80shift'] = {'str':'(mt/1000-64) > 80'}
    dict['metnomu100phPT50'] = {'str' : 'met_nomuon_tst_et > 100000 && trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M && ph_pt[0]>50000'}
    dict['metnomu50phPT200'] = {'str' : 'met_nomuon_tst_et > 50000 && trigger_HLT_g140_loose_L1eEM26M && ph_pt[0]>200000'}
    dict['mtnomu80'] = {'str':'sqrt(2*met_nomuon_tst_et*ph_pt[0]*(1-cos(met_nomuon_tst_phi-ph_phi[0]))) > 80000 '}
    dict['mtnomu80shift'] = {'str':'(sqrt(2*met_nomuon_tst_et*ph_pt[0]*(1-cos(met_nomuon_tst_phi-ph_phi[0])))-64000) > 80000'}

    dict['goodPV'] = {'str' : '(abs(pv_truth_z[0]-pv_z[0]) < 0.5)'}
    dict['wrongPV'] = {'str' : '(abs(pv_truth_z[0]-pv_z[0]) > 0.5)'}
    
    dict['dphiMetPhterm'] = {'str':'acos(cos(met_tst_phi-met_phterm_phi))>=1.25 ', 'shift':'+0'}
    dict['metsig'] = {'str':'met_tst_sig>6 '}
    dict['dmet20'] = {'str':'(met_tst_noJVT_et-met_tst_et)>-20000'}
    dict['dmet10'] = {'str':'(met_tst_noJVT_et-met_tst_et)>-10000'}
    dict['highdmet'] = {'str':'(met_tst_noJVT_et-met_tst_et)>-10000'}
    dict['lowdmet'] = {'str':'(met_tst_noJVT_et-met_tst_et)<-10000'}
    dict['highmetsig'] = {'str':'met_tst_sig > 6'}
    dict['lowjetterm'] = {'str':'met_jetterm_et>100000'}
    dict['highjetterm'] = {'str':'met_jetterm_et<100000'}
    dict['lowmetsig'] = {'str':'met_tst_sig < 6'}
    dict['dmet10phpt'] = {'str':'!((met_tst_noJVT_et-met_tst_et)<-10000 && abs(ph_pt[0]-met_tst_et) < 30000)'}
    dict['dmetNot0'] = {'str':'(met_tst_noJVT_et-met_tst_et)!=0'}
    dict['balance'] = {'str':'(((met_tst_et+ph_pt[0])/(Sum$(jet_central_pt))) <=2) '}
    dict['phEtaLow'] = {'str':'abs(ph_eta[0]) < 1.75'}
    dict['phEtaHigh'] = {'str':'abs(ph_eta[0]) > 1.75'}
    dict['dphiMetJetterm'] = {'str':'((met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0)<0.75'}
    dict['dphiJJ'] = {'str':'(Alt$(acos(cos(jet_central_phi[1]-jet_central_phi[0])),-1))<=2.5'}
    dict['FailVBF'] = {'str':'!((jet_eta[0]*jet_eta[1])<0 && (jet_eta[0]*-jet_eta[1])>3.8)'}
    dict['truthMet'] = {'str':'met_truth_et>75000'}
    dict['failJVTJetPt'] = {'str':'Alt$(failJVT_jet_pt[0], 0) < 35000'}
    dict['BDTScore'] = {'str':'BDTScore > 0.'}
    dict['SR'] = {'str':'n_ph == 1 && n_ph_baseline ==1 && n_mu_baseline == 0 && n_el_baseline == 0 && n_jet<=4 && met_tst_et > 100000 && trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M && ph_pt[0]>50000 && mt >80000 && acos(cos(met_tst_phi-met_phterm_phi))>=1.25 && met_tst_sig>6 && abs(ph_eta[0]) < 1.75 && (met_tst_noJVT_et-met_tst_et)>-10000  && (Alt$(abs(acos(cos(jet_central_phi[1]-jet_central_phi[0]))),0))<=2.5 && ((met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0)<0.75  '}
    dict['2muCR'] = {'str':'trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M & m_uuy > 100000 & n_ph_baseline == 1 && n_mu == 2 && n_el_baseline == 0 && ((ph_topoetcone40[0]-2450.)/ph_pt[0])<=0.022 & met_nomuon_tst_et > 100000 & ph_pt[0]>50000 & sqrt(2*met_nomuon_tst_et*ph_pt[0]*(1-cos(met_nomuon_tst_phi-ph_phi[0]))) > 80000 & acos(cos(met_nomuon_tst_phi-met_phterm_phi)) >= 1.25'}
    dict['Aprime'] = {'str':'abs(ph_eta[0])<1.75 && (met_tst_noJVT_et-met_tst_et)>-10000', 'color': ROOT.kBlue}
    dict['Bprime'] = {'str':'abs(ph_eta[0])<1.75 && (met_tst_noJVT_et-met_tst_et)<-10000', 'color': ROOT.kBlue}
    dict['Cprime'] = {'str':'abs(ph_eta[0])>1.75 && (met_tst_noJVT_et-met_tst_et)>-10000', 'color': ROOT.kBlue}
    dict['Dprime'] = {'str':' abs(ph_eta[0])>1.75 && (met_tst_noJVT_et-met_tst_et)<-10000', 'color': ROOT.kBlue}
    dict['SRC'] = {'str':'n_ph == 1 && n_mu_baseline == 0 && n_el_baseline == 0 && n_jet<=4 && met_tst_et > 100000 && trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M && ph_pt[0]>50000 && mt >80000 && acos(cos(met_tst_phi-met_phterm_phi))>=1.25 && met_tst_sig>6 && abs(ph_eta[0]) > 1.75 && (met_tst_noJVT_et-met_tst_et)>-10000  && ((n_jet_central>=2)*Alt$(abs(acos(cos(jet_central_phi[1]-jet_central_phi[0]))),0)+(n_jet_central<2)*0)<=2.5 && ((met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0)<0.75  '}
    dict['SRBDTscore'] = {'str':'trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M &&n_ph==1 && n_el_baseline==0 && n_mu_baseline==0 && met_tst_et > 100000 && ph_pt[0]>50000&&n_jet<=4 && mt >80000 && BDTScore>=0.1&& acos(cos(met_tst_phi-met_phterm_phi))>=1.&& met_tst_sig > 6 &&((met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0)<=0.75 && Alt$(failJVT_jet_pt[0], 0) < 40000 && Alt$(jet_central_emfrac[0],0) < 0.7 && abs(ph_eta[0]) < 2'}
    dict['baselineBDTscore'] = {'str':'trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M &&n_ph==1 && n_el_baseline==0 && n_mu_baseline==0 && met_tst_et > 100000 && ph_pt[0]>50000&&n_jet<=4 && mt >80000 && BDTScore>=0.1'}
    dict['baseline'] = {'str':'trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M &&n_ph==1 && n_ph_baseline == 1 && n_el_baseline==0 && n_mu_baseline==0 && met_tst_et > 100000 && ph_pt[0]>50000&&n_jet<=4 && mt >80000'}

    #dict['newopt'] = {'str':'BDTScore >= 0.1 && acos(cos(met_tst_phi-met_phterm_phi)) >= 1.25 && met_tst_sig >= 6 && (met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0 < 0.75 && ph_eta[0]<=1.75'}
    dict['newopt'] = {'str': 'acos(cos(met_tst_phi-met_phterm_phi)) >= 1.25 && met_tst_sig >= 6.5 && (met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0 < 0.75 && abs(ph_eta[0])<1.75 && Alt$(failJVT_jet_pt[0], 0) < 45000 && Alt$(jet_central_emfrac[0],0) < 0.7'}
    dict['newoptminus2'] = {'str': 'acos(cos(met_tst_phi-met_phterm_phi)) >= 1.25 && met_tst_sig >= 6.5 && (met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0 < 0.75 && abs(ph_eta[0])<1.75 '}
    dict['BDT01'] = {'str': 'BDTScore >= 0.1'}   
    dict['newoptminus1'] = {'str': 'acos(cos(met_tst_phi-met_phterm_phi)) >= 1.25 && met_tst_sig >= 6.5 && (met_jetterm_et!=0)*Alt$(acos(cos(met_tst_phi-met_jetterm_phi)),0)+(met_jetterm_et==0)*0 < 0.75 && abs(ph_eta[0])<1.75 && Alt$(jet_central_emfrac[0],0) < 0.7'}
 

    return dict
