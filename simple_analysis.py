import ROOT

# Define the path to the directory containing .root files
directory = "/data/tmathew/ntups/mc23d"

# List of root files to process
root_files = ["dijet_y.root",		
              "ggHyyd_y.root",	
              "VHyyd_y.root",   
              "Zgamma_y.root",
              "gammajet_direct_y.root",
              "qqZHyyd_y.root",	
              "Wgamma_y.root",  
              "Zjets_y.root",
              "gammajet_frag_y.root",
              "VBFHyyd_y.root",
              "Wjets_y.root"]

# Histogram configuration
variable = "vtx_sumPt"
xmin, xmax, nbins = 0, 500000, 100  # Define histogram range and bins

# Initialize the histogram
histogram = ROOT.TH1F("vtx_sumPt_hist", f"{variable} Distribution;{variable};Events", nbins, xmin, xmax)

# Loop over the root files
for filename in root_files:
    filepath = f"{directory}/{filename}"
    print(f"Processing file: {filepath}")
    
    # Open the ROOT file
    file = ROOT.TFile.Open(filepath)
    
    # Access the tree named "nominal"
    tree = file.Get("nominal")
    
    if not tree:
        print(f"Could not find TTree 'nominal' in file: {filepath}")
        continue
    
    # Project the TBranch "vtx_sumPt" into the histogram
    tree.Project(histogram.GetName(), variable, f"{variable} >= {xmin} && {variable} <= {xmax}")

# Draw the histogram
canvas = ROOT.TCanvas("canvas", "vtx_sumPt Distribution", 800, 600)
histogram.Draw()

# Save the plot
canvas.SaveAs("vtx_sumPt_distribution.png")

# Keep the canvas open
input("Press Enter to exit.")
