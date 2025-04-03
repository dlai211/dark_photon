
const image1 = [
    'FedericaCut_dphi_met_ph - dphi_met_central_jet_0.png',
    'FedericaCut_dphi_met_ph - dphi_met_central_jet_1.png',
    'FedericaCut_dphi_met_phterm - dphi_met_jetterm_0.png',
    'FedericaCut_dphi_met_phterm - dphi_met_jetterm_1.png',
    'FedericaCutwRunCut_dphi_met_ph - dphi_met_central_jet_0.png',
    'FedericaCutwRunCut_dphi_met_ph - dphi_met_central_jet_1.png',
    'FedericaCutwRunCut_dphi_met_phterm - dphi_met_jetterm_0.png',
    'FedericaCutwRunCut_dphi_met_phterm - dphi_met_jetterm_1.png',
    'SelectionCut_dphi_met_ph - dphi_met_central_jet_0.png',
    'SelectionCut_dphi_met_ph - dphi_met_central_jet_1.png',
    'SelectionCut_dphi_met_phterm - dphi_met_jetterm_0.png',
    'SelectionCut_dphi_met_phterm - dphi_met_jetterm_1.png'
]

const image2 = [

    'ggHyyd_dphi_met_jetterm_vs_dphi_phterm_jetterm_logZ_selection.png',
    'Wgamma_dphi_met_jetterm_vs_dphi_phterm_jetterm_logZ_selection.png',
    'Zgamma_dphi_met_jetterm_vs_dphi_phterm_jetterm_logZ_selection.png',
    'ggHyyd_dphi_met_phterm_vs_dphi_met_jetterm_logZ_selection.png',
    'Wgamma_dphi_met_phterm_vs_dphi_met_jetterm_logZ_selection.png',
    'Zgamma_dphi_met_phterm_vs_dphi_met_jetterm_logZ_selection.png',
    'ggHyyd_dphi_met_phterm_vs_dphi_phterm_jetterm_logZ_selection.png',
    'Wgamma_dphi_met_phterm_vs_dphi_phterm_jetterm_logZ_selection.png',
    'Zgamma_dphi_met_phterm_vs_dphi_phterm_jetterm_logZ_selection.png',
    'dphi_met_jetterm_vs_dphi_phterm_jetterm_scatter_selection.png',
    'dphi_met_phterm_vs_dphi_met_jetterm_scatter_selection.png',
    'dphi_met_phterm_vs_dphi_phterm_jetterm_scatter_selection.png',
    'ggHyyd_dphi_met_jetterm_vs_dphi_phterm_jetterm_logZ_n-2.png',
    'Wgamma_dphi_met_jetterm_vs_dphi_phterm_jetterm_logZ_n-2.png',
    'Zgamma_dphi_met_jetterm_vs_dphi_phterm_jetterm_logZ_n-2.png',
    'ggHyyd_dphi_met_phterm_vs_dphi_met_jetterm_logZ_n-2.png',
    'Wgamma_dphi_met_phterm_vs_dphi_met_jetterm_logZ_n-2.png',
    'Zgamma_dphi_met_phterm_vs_dphi_met_jetterm_logZ_n-2.png',
    'ggHyyd_dphi_met_phterm_vs_dphi_phterm_jetterm_logZ_n-2.png',
    'Wgamma_dphi_met_phterm_vs_dphi_phterm_jetterm_logZ_n-2.png',
    'Zgamma_dphi_met_phterm_vs_dphi_phterm_jetterm_logZ_n-2.png',
    'dphi_met_jetterm_vs_dphi_phterm_jetterm_scatter_n-2.png',
    'dphi_met_phterm_vs_dphi_met_jetterm_scatter_n-2.png',
    'dphi_met_phterm_vs_dphi_phterm_jetterm_scatter_n-2.png',
    'vector_plot_met_ph_jet.png',
    'vector_plot_met_phterm_jetterm.png',
]

const imageMap = {
    "phjet": {
        images: image1,
        path: 'test/wzstudy',
        imagesPerRow: 2, // Number of images to display per row
        title: 'Study of dphi_met_ph - dphi_met_central_jet'
    },
    "2d": {
        images: image2,
        path: 'test/2d_plots',
        imagesPerRow: 3,
        title: '2D plots of dphi_met_phterm, dphi_met_jetterm, dphi_met'
    }
};  