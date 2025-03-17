// Define the cut configurations (same as Python's cut_config)
const cut_config = {
    'basic': true, 'metsig': true, 'dphi_met_phterm': true, 'dmet': true,
    'dphi_met_jetterm': true, 'ph_eta': true, 'dphi_jj': true, 'mt2': true
};

const var_config = [
    "actualIntPerXing", "balance", "balance_sumet", "BDTScore",
    "central_jets_fraction", "dmet", "dphi_jj", "dphi_met_central_jet", 
    "dphi_met_jetterm", "dphi_met_ph", "dphi_met_phterm", "dphi_ph_jet1", 
    "dphi_ph_centraljet1", "dphi_phterm_jetterm", "failJVT_jet_pt", 
    "failJVT_jet_pt1", "goodPV", "jet_central_emfrac", "jet_central_eta", 
    "jet_central_pt", "jet_central_pt1", "jet_central_pt2", "jet_central_timing", 
    "jet_central_timing1", "jetterm", "jetterm_sumet", "met", "met_cst", 
    "met_noJVT", "met_track", "metplusph", "metsig", "metsigres", 
    "mt", "n_jet", "n_jet_central", "n_jet_fwd", "n_el_baseline", 
    "n_mu_baseline", "n_ph", "n_ph_baseline", "n_tau_baseline", "ph_eta", 
    "ph_phi", "ph_pt", "puWeight", "softerm", "trigger", "vtx_sumPt"
];

const sig_config = [
    "BDTScore", "balance", "dmet", "dphi_jj", "dphi_met_jetterm", 
    "dphi_met_phterm", "dphi_ph_centraljet1", "dphi_phterm_jetterm", 
    "met", "metsig", "mt", "n_jet_central", "ph_eta", "ph_pt"
];

// Get references to DOM elements
const ul = document.getElementById("cut-links");
const imageContainer = document.getElementById("image-container");
const cutTitle = document.getElementById("cut-title");
const performanceBtn = document.getElementById("performance-btn");
const significanceBtn = document.getElementById("significance-btn");

let currentMode = null;
let currentCut = null;
// let currentMode = "performance";
// let currentCut = Object.keys(cut_config)[0];

// Function to generate image paths dynamically
function generateImagePaths(cut_name, mode) {
    if (!cut_config[cut_name]) return [];
    
    let images = [];

    if (mode == "performance") {
        var_config.forEach((var_name) => {
            images.push(`mc23d_${cut_name}cut/${var_name}_nodijet.png`);
            images.push(`mc23d_${cut_name}cut/roc_curve_${var_name}.png`);
        })
    } else if (mode == "significance") {
        sig_config.forEach((sig_name) => {
            images.push(`mc23d_${cut_name}cut/${sig_name}_nodijet.png`);
            images.push(`mc23d_${cut_name}cut/significance_${sig_name}_lowercut.png`);
            images.push(`mc23d_${cut_name}cut/significance_${sig_name}_uppercut.png`);
        })
    }

    return images;
}

// Function to update images based on selected cut
function updateImages(cut_name, mode) {
    cutTitle.textContent = `mc23d ${cut_name} cut ${mode} plots`;
    imageContainer.innerHTML = "";

    // Highlight the selected cut name
    document.querySelectorAll("#cut-nav a").forEach(a => a.classList.remove("active"));
    document.querySelectorAll(`a[data-cut='${cut_name}']`).forEach(a => a.classList.add("active"));

    // Determine the number of images per row
    let imagesPerRow = mode === "performance" ? 4 : 3;
    imageContainer.style.display = "grid";
    imageContainer.style.gridTemplateColumns = `repeat(${imagesPerRow}, 1fr)`;

    const images = generateImagePaths(cut_name, mode);
    images.forEach((img) => {
        const container = document.createElement('div');
        container.className = 'image-container';

        const imgElement = document.createElement('img');
        imgElement.src = img;
        imgElement.alt = img.split('/').pop();
        imgElement.onclick = () => openModal(img); // click to zoom

        const filename = document.createElement('p');
        filename.className = 'filename';
        filename.textContent = img.split('/').pop();

        container.appendChild(imgElement);
        container.appendChild(filename);
        imageContainer.appendChild(container);

    })
}

// Create a single nav entry with all cut names, making each cut name clickable
const li = document.createElement("li");
const mainNavText = document.createTextNode("mc23d ");
li.appendChild(mainNavText);

Object.keys(cut_config).forEach((cut, index, array) => {
    // Create clickable cut name
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = cut;
    a.dataset.cut = cut;
    a.onclick = () => {
        currentCut = cut;
        updateImages(currentCut, currentMode);
    }
    a.style.margin = "0 5px"; // Add some spacing

    // Append to the list item
    li.appendChild(a);

    // Add "+" sign if it's not the last element
    if (index < array.length - 1) {
        li.appendChild(document.createTextNode(" + "));
    }

});

li.appendChild(document.createTextNode(" cut"));
// Append the generated nav entry to the list
ul.appendChild(li);



function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll("#mode-nav a").forEach(a => a.classList.remove("active"));

    performanceBtn.classList.toggle("active", mode === "performance");
    significanceBtn.classList.toggle("active", mode === "significance");

    updateImages(currentCut, mode);
}

performanceBtn.onclick = () => switchMode("performance");
significanceBtn.onclick = () => switchMode("significance");


// // Modal Function
// function openModal(src) {
//     document.getElementById("modal-img").src = src;
//     document.getElementById("image-modal").style.display = "flex";
// }

// // Close Modal When Clicking Outside Image
// document.querySelector(".close").onclick = function () {
//     document.getElementById("image-modal").style.display = "none";
// };

// document.getElementById("image-modal").onclick = function (event) {
//     if (event.target === this) {
//         this.style.display = "none";
//     }
// };

function openModal(src) {
    const modal = document.getElementById("image-modal");
    document.getElementById("modal-img").src = src;
    modal.classList.add("show"); // Only add the class now
}

// Close Modal When Clicking Outside Image
document.querySelector(".close").onclick = function () {
    document.getElementById("image-modal").classList.remove("show");
};

document.getElementById("image-modal").onclick = function (event) {
    if (event.target === this) {
        this.classList.remove("show");
    }
};
