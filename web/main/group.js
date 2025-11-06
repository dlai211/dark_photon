
// Helper: detect grouped var_config (array of arrays) and normalize
function getGroupsAndTitles() {
  const groups = Array.isArray(var_config[0]) ? var_config : [var_config];
  const titles = (imageData.group_titles && imageData.group_titles.length === groups.length)
    ? imageData.group_titles
    : groups.map((_, i) => `Group ${i+1}`);
  return { groups, titles };
}

// Helper: build image paths for a list of var names (one group)
function imagePathsForVars(vars, cut_name, mode, lumi) {
  if (!cut_config[cut_name] && mode !== "n-1") return [];
  let base = (lumi === "26fb") ? imageData.path[0] : (lumi === "135fb") ? imageData.path[1] : ``;

  const out = [];
  if (mode === "performance") {
    vars.forEach(v => {
      out.push(base + `mc23e_${cut_name}cut/${v}.png`);
      out.push(base + `mc23e_${cut_name}cut/roc_curve_${v}.png`);
    });
  } else if (mode === "significance") {
    sig_config.forEach(s => {
      out.push(base + `mc23e_${cut_name}cut/${s}.png`);
      out.push(base + `mc23e_${cut_name}cut/significance_${s}_lowercut.png`);
      out.push(base + `mc23e_${cut_name}cut/significance_${s}_uppercut.png`);
    });
  } else if (mode === "n-1") {
    n_1_config.forEach(n1 => {
      out.push(base + `mc23e_n-1cut/${n1}.png`);
      out.push(base + `mc23e_n-1cut/significance_${n1}_lowercut.png`);
      out.push(base + `mc23e_n-1cut/significance_${n1}_uppercut.png`);
    });
  }
  return out;
}

// Build right TOC and render sections per group
function updateImages(cut_name, mode, lumi) {
  cutTitle.textContent = `mc23d & mc23e ${cut_name} cut ${mode} plots`;
  imageContainer.innerHTML = "";

  // Highlight the selected cut name
  document.querySelectorAll("#cut-nav a").forEach(a => a.classList.remove("active"));
  document.querySelectorAll(`a[data-cut='${cut_name}']`).forEach(a => a.classList.add("active"));

  // Number of images per row
  let columns = (mode === "performance") ? 4 : 3;
  imageContainer.style.display = "block"; // parent holds multiple sections now

  const { groups, titles } = getGroupsAndTitles();

  // Build the right-side TOC
  const toc = document.getElementById("group-links");
  if (toc) toc.innerHTML = "";
  groups.forEach((_, i) => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#group-${i}`;
    a.textContent = titles[i];
    a.dataset.target = `group-${i}`;
    // click-to-scroll (smooth via CSS)
    a.onclick = (e) => {
      e.preventDefault();
      document.getElementById(`group-${i}`).scrollIntoView({ behavior: "smooth", block: "start" });
    };
    li.appendChild(a);
    if (toc) toc.appendChild(li);
  });

  // Render each group as its own section
  groups.forEach((vars, i) => {
    const section = document.createElement("section");
    section.className = "group-section";
    section.id = `group-${i}`;

    const h = document.createElement("h3");
    h.textContent = titles[i];
    section.appendChild(h);

    // grid container per group
    const grid = document.createElement("div");
    grid.className = "wrapper";
    grid.style.display = "grid";
    grid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
    grid.style.gap = "10px";

    const imgs = imagePathsForVars(vars, cut_name, mode, lumi);
    imgs.forEach((img) => {
      const container = document.createElement('div');
      container.className = 'image-container';

      const imgElement = document.createElement('img');
      imgElement.src = img;
      imgElement.alt = img.split('/').pop();
      imgElement.onclick = () => openModal(img);

      const filename = document.createElement('p');
      filename.className = 'filename';
      filename.textContent = img.split('/').pop();

      container.appendChild(imgElement);
      container.appendChild(filename);
      grid.appendChild(container);
    });

    section.appendChild(grid);
    imageContainer.appendChild(section);
  });

  // Auto-highlight current group in TOC while scrolling
  if (typeof IntersectionObserver !== "undefined") {
    const links = document.querySelectorAll("#group-toc a");
    const sections = [...document.querySelectorAll(".group-section")];
    const obs = new IntersectionObserver((entries) => {
      // find the most visible section
      let topMost = null, topY = Infinity;
      entries.forEach(e => {
        if (e.isIntersecting) {
          const y = e.target.getBoundingClientRect().top;
          if (y >= 0 && y < topY) { topY = y; topMost = e.target; }
        }
      });
      if (topMost) {
        const id = topMost.id;
        links.forEach(a => a.classList.toggle("active", a.dataset.target === id));
      }
    }, { rootMargin: "-20% 0px -60% 0px", threshold: [0, 0.2, 0.6, 1] });

    sections.forEach(s => obs.observe(s));
  }
}
