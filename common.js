// different config.js based on the link
const hash = window.location.hash.replace('#', '');
console.log("Hash from URL: ", hash);

// Get references to DOM elements
const imageContainer = document.getElementById("image-container");
const cutTitle = document.getElementById("cut-title");
const imageData = imageMap[hash] || { images: [], title: 'No plots found' };
let imagesPerRow = imageData.imagesPerRow || 2; // Default to 2 if not set

// Function to update images based on selected cut
function updateImages() {
    imageContainer.innerHTML = "";
    cutTitle.textContent = imageData.title;

    imageContainer.style.display = "grid";
    imageContainer.style.gridTemplateColumns = `repeat(imagesPerRow, 1fr)`;

    const images = imageData.images.map(file => `${imageData.path}/${file}`);
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

updateImages();

// Change the images per row based on the mode selected 
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        imagesPerRow = parseInt(btn.textContent);
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateImages();
    });
});

// // Modal Function
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
