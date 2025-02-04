#include <iostream>
#include <vector>
#include <string>
#include <chrono>  // For timing
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Define the path to the directory containing .root files
    std::string directory = "/data/tmathew/ntups/mc23d";

    // List of ROOT files to process
    std::vector<std::string> root_files = {
        "dijet_y.root", "ggHyyd_y.root", "VHyyd_y.root",
        "Zgamma_y.root", "gammajet_direct_y.root", "qqZHyyd_y.root",
        "Wgamma_y.root", "Zjets_y.root", "gammajet_frag_y.root",
        "VBFHyyd_y.root", "Wjets_y.root"
    };

    // Histogram configuration
    std::string variable = "vtx_sumPt";
    float xmin = 0, xmax = 500000;
    int nbins = 100;

    // Initialize the histogram
    TH1F *histogram = new TH1F("vtx_sumPt_hist", (variable + " Distribution;" + variable + ";Events").c_str(), nbins, xmin, xmax);

    // Loop over the ROOT files
    for (const auto& filename : root_files) {
        std::string filepath = directory + "/" + filename;
        std::cout << "Processing file: " << filepath << std::endl;

        // Open the ROOT file
        TFile *file = TFile::Open(filepath.c_str());
        if (!file || file->IsZombie()) {
            std::cerr << "Could not open file: " << filepath << std::endl;
            continue;
        }

        // Access the tree named "nominal"
        TTree *tree = (TTree*)file->Get("nominal");
        if (!tree) {
            std::cerr << "Could not find TTree 'nominal' in file: " << filepath << std::endl;
            file->Close();
            continue;
        }

        // Project the TBranch "vtx_sumPt" into the histogram
        tree->Project(histogram->GetName(), variable.c_str(), (variable + " >= " + std::to_string(xmin) + " && " + variable + " <= " + std::to_string(xmax)).c_str());

        file->Close();
    }

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Draw the histogram
    TCanvas *canvas = new TCanvas("canvas", "vtx_sumPt Distribution", 800, 600);
    histogram->Draw();

    // Save the plot
    canvas->SaveAs("vtx_sumPt_distribution.png");

    // Print execution time
    std::cout << "\nExecution Time: " << elapsed_time.count() << " seconds" << std::endl;

    // Keep the canvas open
    std::cout << "Press Enter to exit...";
    std::cin.get();

    delete canvas;
    delete histogram;

    return 0;
}
