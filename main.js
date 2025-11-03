
// Get references to DOM elements
const ul = document.getElementById("cut-links");
const imageContainer = document.getElementById("image-container");
const cutTitle = document.getElementById("cut-title");
const performanceBtn = document.getElementById("performance-btn");
const significanceBtn = document.getElementById("significance-btn");
const n1Btn = document.getElementById("n-1-btn");

let currentMode = null;
let currentCut = null;
let currentLumi = "135fb";
// let currentMode = "performance";
// let currentCut = Object.keys(cut_config)[0];

// Create a single nav entry with all cut names, making each cut name clickable
const li = document.createElement("li");
const mainNavText = document.createTextNode("mc23e ");
li.appendChild(mainNavText);

Object.keys(cut_config).forEach((cut, index, array) => {
    // Create clickable cut name
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = cut;
    a.dataset.cut = cut;
    a.onclick = () => {
        currentCut = cut;
        updateImages(currentCut, currentMode, currentLumi);
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

    document.querySelectorAll("#mode-nav a.mode-btn").forEach(a => a.classList.remove("active"));
    // document.getElementById(mode === "performance" ? "performance-btn" : "significance-btn").classList.add("active");

    performanceBtn.classList.toggle("active", mode === "performance");
    significanceBtn.classList.toggle("active", mode === "significance");
    n1Btn.classList.toggle("active", mode === "n-1");

    updateImages(currentCut, currentMode, currentLumi);
}

function switchLumi(lumi) {
    currentLumi = lumi;

    document.querySelectorAll("#mode-nav a.lumi-btn").forEach(a => a.classList.remove("active"));
    document.getElementById(lumi === "26fb" ? "lumi-26-btn" : "lumi-135-btn").classList.add("active");

    updateImages(currentCut, currentMode, currentLumi);
}

performanceBtn.onclick = () => switchMode("performance");
significanceBtn.onclick = () => switchMode("significance");
n1Btn.onclick = () => switchMode("n-1");
document.getElementById("lumi-26-btn").onclick = () => switchLumi("26fb");
document.getElementById("lumi-135-btn").onclick = () => switchLumi("135fb");

switchLumi("135fb");

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
