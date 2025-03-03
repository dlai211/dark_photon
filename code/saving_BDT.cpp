#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <map>

void AddBDTScoreToNtuple() {
    // Paths to input and output files
    TString input_file_path = "/data/jlai/ntups/mc23d_tmp/mc23d_ggHyyd_y_BDT_score.root";  // File with BDTScore
    TString target_file_path = "/data/jlai/ntups/mc23d_tmp/ggHyyd_y.root";  // Main ntuple
    TString output_file_path = "/data/jlai/ntups/mc23d_tmp/ggHyyd_y_with_BDTScore.root";  // New file with BDTScore added

    // Open the input ROOT files
    TFile *input_file = TFile::Open(input_file_path, "READ");
    if (!input_file || input_file->IsZombie()) {
        std::cerr << "ERROR: Could not open " << input_file_path << std::endl;
        return;
    }
    TTree *input_tree = (TTree*) input_file->Get("nominal");

    TFile *target_file = TFile::Open(target_file_path, "READ");
    if (!target_file || target_file->IsZombie()) {
        std::cerr << "ERROR: Could not open " << target_file_path << std::endl;
        return;
    }
    TTree *target_tree = (TTree*) target_file->Get("nominal");

    // Create the output file
    TFile *output_file = new TFile(output_file_path, "RECREATE");
    if (!output_file || output_file->IsZombie()) {
        std::cerr << "ERROR: Could not create " << output_file_path << std::endl;
        return;
    }

    // Clone the target tree structure
    TTree *output_tree = target_tree->CloneTree(0);

    // Variables to store data
    ULong64_t event_number;
    float BDTScore;
    input_tree->SetBranchAddress("event", &event_number);
    input_tree->SetBranchAddress("BDTScore", &BDTScore);

    ULong64_t target_event;
    float new_BDTScore = -999;
    target_tree->SetBranchAddress("event", &target_event);
    TBranch *newBranch = output_tree->Branch("BDTScore", &new_BDTScore, "BDTScore/F");

    // Process event-by-event instead of using a large map
    for (ULong64_t i = 0; i < target_tree->GetEntries(); i++) {
        target_tree->GetEntry(i);
        new_BDTScore = -999;  // Default value

        for (ULong64_t j = 0; j < input_tree->GetEntries(); j++) {
            input_tree->GetEntry(j);
            if (event_number == target_event) {
                new_BDTScore = BDTScore;
                break;  // Stop searching once a match is found
            }
        }

        output_tree->Fill();
    }

    // Save and close files
    output_tree->Write();
    output_file->Close();
    target_file->Close();
    input_file->Close();

    std::cout << "✅ New ROOT file saved at: " << output_file_path << std::endl;
}

// **Add this main function**
int main() {
    AddBDTScoreToNtuple();
    return 0;
}
