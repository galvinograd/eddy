(() => {
  const SLIDES = [
    { id: '00010', prompt: 'A young boy in his pajamas is playing a video game.',
      iid: 'gallery/none', eddy: 'gallery/eddy_bw0_013' },
    { id: '00007', prompt: 'A woman writing something down on paper while the laptop sits on the table.',
      iid: 'gallery/none_pp3', eddy: 'gallery/eddy_bw0_016_pp3' },
    { id: '00001', prompt: 'A man on a motorbike rides down the street.',
      iid: 'gallery/none_sdxl_pp2', eddy: 'gallery/eddy_sdxl_pp2' },
    { id: '00000', prompt: 'No one is in the room but there are chairs.',
      iid: 'gallery/none_pp3', eddy: 'gallery/eddy_bw0_016_pp3' },
    { id: '00002', prompt: 'Numerous motor scooters parked by backing in facing the street.',
      iid: 'gallery/none_sdxl_pp3', eddy: 'gallery/eddy_sdxl_pp3' },
    { id: '00005', prompt: 'A plate of stir fry vegetables on white rice.',
      iid: 'gallery/none_pp3', eddy: 'gallery/eddy_bw0_016_pp3' },
    { id: '00018', prompt: 'A woman sitting in front of a giant pizza.',
      iid: 'gallery/none_pp1', eddy: 'gallery/eddy_bw0_013_pp1' },
    { id: '00016', prompt: 'A person on skis on snowy forest path.',
      iid: 'gallery/none_pp3', eddy: 'gallery/eddy_bw0_016_pp3' },
    { id: '00004', prompt: 'Many electronic devices and a tangle of wires are on and around the desk.',
      iid: 'gallery/none_pp3', eddy: 'gallery/eddy_bw0_016_pp3' },
  ];

  const INTERVAL = 4500;
  const SLIDE_GAP = 20;
  const TRANSITION_MS = 400;
  const N = SLIDES.length;
  // cur: 0 = clone of last, 1..N = real slides, N+1 = clone of first
  let cur = 1, timer, paused = false, track;

  function slidePos(idx) {
    return `translateX(calc(-${idx * 100}% - ${idx * SLIDE_GAP}px))`;
  }

  function imgGrid(dir, id, eager) {
    return [0, 1, 2, 3].map(v =>
      `<div class="img-wrap"><img src="${dir}/img_${id}_v${v}.webp" alt="variant ${v}"${eager ? '' : ' loading="lazy"'} decoding="async" onload="this.parentElement.classList.add('loaded')" onerror="this.parentElement.classList.add('loaded')"></div>`
    ).join('');
  }

  function slideHtml(s, eager) {
    return `<div class="gallery-slide">
  <div class="gallery-cols">
    <div class="gallery-col">
      <div class="gallery-col-label">I.I.D</div>
      <div class="gallery-img-grid">${imgGrid(s.iid, s.id, eager)}</div>
    </div>
    <div class="gallery-col">
      <div class="gallery-col-label">EDDY <span class="muted">(Ours)</span></div>
      <div class="gallery-img-grid">${imgGrid(s.eddy, s.id, eager)}</div>
    </div>
  </div>
  <div class="gallery-meta">
    <span class="gallery-prompt">&ldquo;${s.prompt}&rdquo;</span>
  </div>
</div>`;
  }

  function render(viewportEl, dotsEl) {
    track = document.createElement('div');
    track.className = 'gallery-track';
    track.style.gap = `${SLIDE_GAP}px`;
    // Clone first and last slides so boundary swipes reveal a neighbour
    track.innerHTML =
      slideHtml(SLIDES[N - 1], false) +
      SLIDES.map((s, i) => slideHtml(s, i === 0)).join('') +
      slideHtml(SLIDES[0], false);
    track.style.transform = slidePos(1);
    viewportEl.innerHTML = '';
    viewportEl.appendChild(track);

    dotsEl.innerHTML = SLIDES.map((_, i) =>
      `<button class="gallery-dot${i === 0 ? ' active' : ''}" data-idx="${i + 1}" aria-label="Slide ${i + 1}"></button>`
    ).join('');
  }

  function updateDots(pos) {
    const realIdx = ((pos - 1 + N) % N);
    document.querySelectorAll('.gallery-dot').forEach((d, i) => {
      d.classList.toggle('active', i === realIdx);
    });
  }

  function setPos(pos) {
    track.style.transition = 'none';
    track.getBoundingClientRect();
    track.style.transform = slidePos(pos);
    cur = pos;
    // Double RAF: the callback queued inside a RAF runs in the next frame,
    // so Paint 1 commits the instant jump before Paint 2 re-enables the transition.
    requestAnimationFrame(() => requestAnimationFrame(() => { track.style.transition = ''; }));
  }

  function goTo(pos) {
    updateDots(pos);
    cur = pos;
    track.style.transform = slidePos(pos);
    if (pos === 0 || pos === N + 1) {
      setTimeout(() => setPos(pos === 0 ? N : 1), TRANSITION_MS);
    }
  }

  function resetTimer() {
    clearInterval(timer);
    timer = setInterval(() => { if (!paused) goTo(cur + 1); }, INTERVAL);
  }

  document.addEventListener('DOMContentLoaded', () => {
    const viewportEl = document.getElementById('gallerySlides');
    const dotsEl     = document.getElementById('galleryDots');
    const wrap       = document.getElementById('galleryWrap');
    if (!viewportEl) return;

    render(viewportEl, dotsEl);

    document.getElementById('galleryPrev').addEventListener('click', () => { goTo(cur - 1); resetTimer(); });
    document.getElementById('galleryNext').addEventListener('click', () => { goTo(cur + 1); resetTimer(); });

    dotsEl.addEventListener('click', e => {
      const dot = e.target.closest('.gallery-dot');
      if (dot) { goTo(+dot.dataset.idx); resetTimer(); }
    });

    wrap.addEventListener('mouseenter', () => { paused = true; });
    wrap.addEventListener('mouseleave', () => { paused = false; });

    let touchStartX = 0;
    wrap.addEventListener('touchstart', e => {
      touchStartX = e.touches[0].clientX;
      track.style.transition = 'none';
    }, { passive: true });
    wrap.addEventListener('touchmove', e => {
      const dx = e.touches[0].clientX - touchStartX;
      track.style.transform = `translateX(calc(-${cur * 100}% - ${cur * SLIDE_GAP}px + ${dx}px))`;
    }, { passive: true });
    wrap.addEventListener('touchend', e => {
      const dx = e.changedTouches[0].clientX - touchStartX;
      track.style.transition = '';
      track.getBoundingClientRect();
      if (Math.abs(dx) > 50) { goTo(cur + (dx < 0 ? 1 : -1)); resetTimer(); }
      else { track.style.transform = slidePos(cur); }
    }, { passive: true });

    resetTimer();
  });
})();
