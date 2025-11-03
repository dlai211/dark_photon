// different config.js based on the link
let hash = window.location.hash.replace('#', '');
console.log("Hash from URL: ", hash);
if (!hash) {hash = "main"};

// Get imageData dict
const imageData = imageMap_index[hash] || { images: [], title: 'No plots found' };
let imagesPerRow = imageData.imagesPerRow || 4; // Default to 4 if not set
const cut_config = imageData.cut_config;
const var_config = imageData.var_config;
const sig_config = imageData.sig_config;
const n_1_config = imageData.n_1_config;

// Function to generate image paths dynamically
function generateImagePaths(cut_name, mode, lumi) {
    if (!cut_config[cut_name] && mode !== "n-1") return [];
    
    let images = [];
    let path = (lumi === "26fb") ? imageData.path[0] : (lumi === "135fb") ? imageData.path[1] : ``;

    if (mode == "performance") {
        var_config.forEach((var_name) => {
            images.push(path + `mc23e_${cut_name}cut/${var_name}.png`);
            images.push(path + `mc23e_${cut_name}cut/roc_curve_${var_name}.png`);
        })
    } else if (mode == "significance") {
        sig_config.forEach((sig_name) => {
            images.push(path + `mc23e_${cut_name}cut/${sig_name}.png`);
            images.push(path + `mc23e_${cut_name}cut/significance_${sig_name}_lowercut.png`);
            images.push(path + `mc23e_${cut_name}cut/significance_${sig_name}_uppercut.png`);
        })
    } else if (mode == "n-1") {
        n_1_config.forEach((n_1_name) => {
            images.push(path + `mc23e_n-1cut/${n_1_name}.png`);
            images.push(path + `mc23e_n-1cut/significance_${n_1_name}_lowercut.png`);
            images.push(path + `mc23e_n-1cut/significance_${n_1_name}_uppercut.png`);
        })
    }

    return images;
}

// Function to update images based on selected cut
function updateImages(cut_name, mode, lumi) {
    cutTitle.textContent = `mc23d & mc23e ${cut_name} cut ${mode} plots`;
    imageContainer.innerHTML = "";

    // Highlight the selected cut name
    document.querySelectorAll("#cut-nav a").forEach(a => a.classList.remove("active"));
    document.querySelectorAll(`a[data-cut='${cut_name}']`).forEach(a => a.classList.add("active"));

    // Determine the number of images per row
    let imagesPerRow = mode === "performance" ? 4 : 3;
    imageContainer.style.display = "grid";
    imageContainer.style.gridTemplateColumns = `repeat(${imagesPerRow}, 1fr)`;

    const images = generateImagePaths(cut_name, mode, lumi);
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