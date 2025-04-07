
# include <TFile.h>
# include <TTree.h>
# include <TString.h>
# include <TMVA/Factory.h>
# include <TMVA/Tools.h>
# include <TMVA/Reader.h>
# include <TMVA/DataLoader.h>
# include <iostream>
using namespace std;

int main() {

    TMVA::Tools::Instance();
    
    TFile *input_sig = TFile::Open("/data/jlai/ntups/mc23d/tmva_input_signal.root");
    TFile *input_bkg = TFile::Open("/data/jlai/ntups/mc23d/tmva_input_background.root");

    TTree *tsig = (TTree*)input_sig->Get("nominal");
    TTree *tbkg = (TTree*)input_bkg->Get("nominal");

    if (!tsig || !tbkg) {
        std::cerr << "ERROR: Tree not found." << std::endl;
        return 1;
    }

    std::cout << "Signal entries: " << tsig->GetEntries() << std::endl;
    std::cout << "Background entries: " << tbkg->GetEntries() << std::endl;


    TFile *outputFile = TFile::Open("/data/jlai/ntups/mc23d/tmva_output.root", "RECREATE");

    // !V -> Verbose mode disabled
    // !Silent -> Batch mode disabled
    // Transofmrations on input variables -> Identity (I), Descorrletation (D), PCA (P), Uniform(U), Gaussian (G)
    // AnalysisType -> Classification, Regression, Multiclass, Auto
    TMVA::Factory* factory = new TMVA::Factory("TMVAClassification", outputFile,
        "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification:Transformations=I");

    TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

    dataloader->AddVariable("metsig", 'F');
    dataloader->AddVariable("metsigres", 'F');
    dataloader->AddVariable("met", 'F');
    dataloader->AddVariable("met_noJVT", 'F');
    dataloader->AddVariable("dmet", 'F');
    dataloader->AddVariable("ph_pt", 'F');
    dataloader->AddVariable("ph_eta", 'F');
    dataloader->AddVariable("ph_phi", 'F');
    dataloader->AddVariable("jet_central_eta", 'F');
    dataloader->AddVariable("jet_central_pt1", 'F');
    dataloader->AddVariable("jet_central_pt2", 'F');
    dataloader->AddVariable("dphi_met_phterm", 'F');
    dataloader->AddVariable("dphi_met_ph", 'F');
    dataloader->AddVariable("dphi_met_jetterm", 'F');
    dataloader->AddVariable("dphi_phterm_jetterm", 'F');
    dataloader->AddVariable("dphi_ph_centraljet1", 'F');
    dataloader->AddVariable("metplusph", 'F');
    dataloader->AddVariable("failJVT_jet_pt1", 'F');
    dataloader->AddVariable("softerm", 'F');
    dataloader->AddVariable("jetterm", 'F');
    dataloader->AddVariable("n_jet_central", 'F'); 
    dataloader->AddVariable("jetterm_sumet", 'F');
    dataloader->AddVariable("dphi_met_central_jet", 'F');
    dataloader->AddVariable("balance", 'F');
    dataloader->AddVariable("dphi_jj", 'F');
    // dataloader->AddVariable("BDTScore", 'F');


    // // Assign global weights per tree
    double_t signalWeight = 1.0;
    double_t backgroundWeight = 1.0;
    dataloader->AddSignalTree (tsig, signalWeight);
    dataloader->AddBackgroundTree(tbkg, backgroundWeight);

    // // Set individual even weights (variables must exist in the original TTree)
    // dataloader->SetSignalWeightExpression("weight");
    // dataloader->SetBackgroundWeightExpression("weight");

    dataloader->SetWeightExpression("weights");

    // Apply additional cuts on the samples
    TCut mycuts = "metsig<=16";
    TCut mycutb = "metsig<=16";

    // Creating Training and Test Trees
    // If nTrain_Signal and nTrain_Background are both 0, the total sample is split in half for training and testing.
    // NormMode -> Overall renormalization of event-by-event weights used in the training
        // None: No normalization
        // NumEvents: average weight of 1 per event, independently for Signal and Background
        // EqualNumEvents: average weight of 1 per event for signal and sum of weights for background equal to sum of weights for signal
    dataloader->PrepareTrainingAndTestTree(
        mycuts, mycutb, 
        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=EqualNumEvents:!V"
    );

    // Classification
    // factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
    //     "!H:!V:NTrees=200:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:"
    //     "UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20");

    std::vector<int> depths = {2, 3, 4, 5}; // Different depths for BDT
    std::vector<int> nTrees = {100, 200, 300}; // Different number of trees for BDT
    for (int depth: depths) {
        for (int ntree: nTrees) {
            TString methodName = Form("BDT_Depth%d_Trees%d", depth, ntree);
            TString options = Form(
                "!H:!V:NTrees=%d:MaxDepth=%d:BoostType=AdaBoost:AdaBoostBeta=0.5:"
                "UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20",
                ntree, depth);
            factory->BookMethod(dataloader, TMVA::Types::kBDT, methodName.Data(), options);
            std::cout << "Booked method: " << methodName.Data() << " with options: " << options.Data() << std::endl;
        }
    }

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    outputFile->Close();

    std::cout << "==> TMVA training complete. Output written to TMVA_output.root" << std::endl;

    return 0;
}