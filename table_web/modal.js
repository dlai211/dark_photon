(() => {
  // --- Image hover tooltip (uses data-img if present) ---
  let hoverTip = null;
  const mkHover = (target) => {
    const imgSrc = target.getAttribute('data-img');
    if (!imgSrc) return;

    hoverTip = document.createElement('div');
    hoverTip.className = 'img-tip';
    // optional: include range text (data-tip) under the image
    const cap = target.getAttribute('data-tip') || '';
    hoverTip.innerHTML = `
      <img src="${imgSrc}" alt="definition">
      ${cap ? `<div class="img-tip-cap">${cap}</div>` : ''}
    `;
    document.body.appendChild(hoverTip);

    const move = (e) => {
      const pad = 12;
      hoverTip.style.left = (e.pageX + pad) + 'px';
      hoverTip.style.top  = (e.pageY + pad) + 'px';
    };
    target.addEventListener('mousemove', move, { passive: true });
    hoverTip._cleanup = () => target.removeEventListener('mousemove', move);
  };

  const killHover = () => {
    if (hoverTip) {
      hoverTip._cleanup?.();
      hoverTip.remove();
      hoverTip = null;
    }
  };

  // Attach to .var elements
  document.querySelectorAll('.var').forEach(el => {
    el.addEventListener('mouseenter', () => mkHover(el));
    el.addEventListener('mouseleave', killHover);
    el.addEventListener('click', (e) => {
      const imgSrc = el.getAttribute('data-img');
      if (!imgSrc) return;  // only image-enabled vars open modal
      e.preventDefault();
      openModal(imgSrc, el.textContent.trim());
    });
  });

  // --- Modal logic ---
  const modal = document.getElementById('var-modal');
  const modalImg = document.getElementById('var-modal-img');
  const modalCap = document.getElementById('var-modal-cap');

  function openModal(src, title = '') {
    modalImg.src = src;
    modalCap.textContent = title || '';
    modal.setAttribute('aria-hidden', 'false');
    document.documentElement.classList.add('no-scroll');
  }

  function closeModal() {
    modal.setAttribute('aria-hidden', 'true');
    modalImg.removeAttribute('src');
    document.documentElement.classList.remove('no-scroll');
  }

  modal.addEventListener('click', (e) => {
    if (e.target.closest('[data-close]')) closeModal();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.getAttribute('aria-hidden') === 'false') {
      closeModal();
    }
  });
})();
