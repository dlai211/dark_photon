from __future__ import print_function

import ROOT, sys, os

def parse_CSV_file_with_TTree_ReadStream():

    ROOT.gROOT.SetBatch(True)  # Run in batch mode to avoid GUI issues
    header_mapping_dictionary = {
        'metsig'                : ('metsig',               float),
        'metsigres'             : ('metsigres',             float),
        'met'                   : ('met',                   float),
        'met_noJVT'             : ('met_noJVT',             float),
        'dmet'                  : ('dmet',                  float),
        'ph_pt'                 : ('ph_pt',                 float),
        'ph_eta'                : ('ph_eta',                float),
        'ph_phi'                : ('ph_phi',                float),
        'jet_central_eta'       : ('jet_central_eta',       float),
        'jet_central_pt1'       : ('jet_central_pt1',       float),
        'jet_central_pt2'       : ('jet_central_pt2',       float),
        'dphi_met_phterm'       : ('dphi_met_phterm',       float),
        'dphi_met_ph'           : ('dphi_met_ph',           float),
        'dphi_met_jetterm'      : ('dphi_met_jetterm',      float),
        'dphi_phterm_jetterm'   : ('dphi_phterm_jetterm',   float),
        'dphi_ph_centraljet1'   : ('dphi_ph_centraljet1',   float),
        'metplusph'             : ('metplusph',             float),
        'failJVT_jet_pt1'       : ('failJVT_jet_pt1',       float),
        'softerm'               : ('softerm',               float),
        'jetterm'               : ('jetterm',               float),
        'jetterm_sumet'         : ('jetterm_sumet',         float),
        'n_jet_central'         : ('n_jet_central',         int),
        'dphi_met_central_jet'  : ('dphi_met_central_jet',  float),
        'balance'               : ('balance',               float),
        'dphi_jj'               : ('dphi_jj',               float),
        'BDTScore'              : ('BDTScore',              float),
        'mt'                    : ('mt',                    float),
        'weights'               : ('weights',               float),
        'process'               : ('process',               str), 
        'label'                 : ('label',                 int)
    }

    type_mapping_dictionary = {
        str:    'C', 
        int :   'I',
        float : 'F',
    }
    
    tree_name = "nominal"
    csv_file_name = "/data/jlai/dark_photon/code2/BDT_input_background.csv"
    header_row = open(csv_file_name).readline().strip().split(',')
    print(header_row)

    # Create the branch descriptor
    branch_descriptor = ':'.join([header_mapping_dictionary[row][0]+'/'+
                           type_mapping_dictionary[header_mapping_dictionary[row][1]]
                           for row in header_row if row in header_mapping_dictionary])

    output_ROOT_file_name = "/data/jlai/ntups/mc23d/tmva_input_background.root"
    output_file = ROOT.TFile(output_ROOT_file_name, "recreate")
    print("Outputting %s -> %s" % (csv_file_name, output_ROOT_file_name))

    output_tree = ROOT.TTree(tree_name, tree_name)
    file_lines = open(csv_file_name).readlines()

    # Clean the data entries: remove the first (header) row.
    # Ensure empty strings are tagged as such since ROOT doesn't differentiate between different types
    # of white space.  Therefore, we change all of these entries to 'empty'. 
    # Also, avoiding any lines that begin with '#'
    file_lines     = ['\t'.join([val if (val.find(' ') == -1 and val != '')
                                else 'empty' for val in line.split(',')])
                             for line in file_lines[1:] if line[0] != '#' ]
 
    # Removing NaN, setting these entries to 0.0.
    # Also joining the list of strings into one large string.
    file_as_string = ('\n'.join(file_lines)).replace('NaN', str(0.0))

    istring = ROOT.istringstream(file_as_string)

    output_tree.ReadStream(istring, branch_descriptor)

    output_file.cd()
    output_tree.Write()



if __name__ == "__main__":
    parse_CSV_file_with_TTree_ReadStream()
