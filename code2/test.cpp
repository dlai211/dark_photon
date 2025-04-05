#include <TFile.h>
#include <TTree.h>
#include <TMVA/Factory.h>
#include <TMVA/Tools.h>
#include <TMVA/DataLoader.h>
#include <iostream>

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

    TFile *outputFile = TFile::Open("tmva_output.root", "RECREATE");
    TMVA::Factory *factory = new TMVA::Factory("TMVAClassification", outputFile,
        "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification:Transformations=I");

    TMVA::DataLoader *loader = new TMVA::DataLoader("dataset");

    // Use only 2 variables
    loader->AddVariable("met", 'F');
    loader->AddVariable("ph_pt", 'F');

    loader->AddSignalTree(tsig, 1.0);
    loader->AddBackgroundTree(tbkg, 1.0);

    loader->SetWeightExpression("weights");

    TCut cut = "met > 0";  // basic sanity cut
    loader->PrepareTrainingAndTestTree(cut, cut, 
        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V");

    factory->BookMethod(loader, TMVA::Types::kBDT, "BDT",
        "!H:!V:NTrees=200:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:"
        "UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:"
        "IgnoreNegWeightsInTraining=True:nCuts=20");
        

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    outputFile->Close();
    std::cout << "DONE." << std::endl;

    return 0;
}
