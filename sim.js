/* EDDY page — particle visualisation that respects the analytical p_t.
 *
 * Setup (matches the paper, Sec. 4.2 / Eq. (40)):
 *   - 2D, target marginal p = N(0, I_2)  =>  score s(x) = -x
 *   - One source x_src is held fixed; v = -x_src / ||x_src||  (unit, → origin)
 *   - delta = x - x_src,  k = exp(-||delta||^2 / gamma)
 *
 *   PG:        psi_PG(x)   = (2/gamma) * k * delta
 *   EDDY-RBF:  psi_EDDY(x) = (2/gamma) * k * ( C_delta * delta  +  C_v * v )
 *     C_delta = (2/gamma) <delta, v> + <v, s>
 *     C_v     = d - 1 - (2/gamma) ||delta||^2 - <delta, s>
 *
 * The marginal of  dx = mu dt + sqrt(2) dW,  mu = -x + w_g * psi,  is known
 * in closed form:
 *
 *   EDDY-RBF  (Claim 1):  p_EDDY(x, t)  =  N(0, I_2)(x)        for all t
 *
 *   PG  (Gibbs, since mu_PG = -grad U):
 *                          p_PG^infty(x) ∝ N(0, I_2)(x) * exp(-w_g * k(x, x_src))
 *
 * We visualise these by running a particle pool that *spawns from the
 * analytical p_t* and then moves only under psi (no OU drift, no diffusion).
 * Birth/death keeps the empirical distribution pinned to p_t — by Claim 1
 * for EDDY (psi_EDDY is FPE-divergence-free w.r.t. N(0, I_2), so flow alone
 * preserves it exactly) and by short lifespans for PG (the spawn term
 * dominates so the cloud stays near p_PG^infty).
 *
 * Each particle has an age. New particles are sampled from p_t. They
 * evolve under  dx/dt = w_g * psi  (RK4) until they age out, at which point
 * they respawn from p_t. Per-particle alpha is multiplied by a sin life
 * envelope so they fade in / live / fade out smoothly.
 */

(() => {
  // -------- Tunables --------
  const GAMMA       = 0.5;
  const W_G         = 10.0;
  const D           = 2;
  const VIEW_HALF   = 3.5;          // world coords: [-VIEW_HALF, VIEW_HALF]
  const STAR_RADIUS = 10;

  // Particle pool
  const N_PARTICLES     = 5_000;
  const LIFESPAN_FRAMES = 120;       // ~0.67 s @ 60 fps  — short keeps cloud at p_t
  const PARTICLE_BASE_A = 0.025;     // peak alpha at mid-life
  const DOT_RADIUS      = 6;
  const SIM_DT          = 0.005;    // per sub-step
  const SUB_STEPS       = 1;        // sub-steps per frame

  // Rejection-sampling safety: max attempts before falling back to N(0, I).
  const REJ_MAX_TRIES   = 200;

  // -------- Random / sampling --------
  const randn = () => {
    const u1 = Math.random() || 1e-12;
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  // Sample one (x, y) from the analytical p_t for the given panel kind.
  //   "EDDY":  p_t = N(0, I_2)                                 (direct)
  //   "PG":    p ∝ N(0, I_2) * exp(-w_g * k(x, x_src))         (rejection)
  // For PG we draw a candidate from N(0, I_2) and accept with probability
  // exp(-w_g * k); the proposal envelope is exact (PG's density is the
  // proposal × that acceptance), so the accepted sample is iid p_PG^infty.
  function sampleFromP(kind, srcX, srcY, out) {
    if (kind === "EDDY") {
      out[0] = randn();
      out[1] = randn();
      return;
    }
    // PG
    const invG = 1 / GAMMA;
    for (let n = 0; n < REJ_MAX_TRIES; n++) {
      const x = randn(), y = randn();
      const dx = x - srcX, dy = y - srcY;
      const k = Math.exp(-(dx * dx + dy * dy) * invG);
      if (Math.random() < Math.exp(-W_G * k)) {
        out[0] = x; out[1] = y;
        return;
      }
    }
    // Fallback (statistically negligible for the parameter range we use).
    out[0] = randn();
    out[1] = randn();
  }

  // -------- Fields --------
  function fieldPG(px, py, sx, sy, _vx, _vy) {
    const dx = px - sx, dy = py - sy;
    const k  = Math.exp(-(dx * dx + dy * dy) / GAMMA);
    const c  = (2 / GAMMA) * k;
    return [c * dx, c * dy];
  }

  function fieldEDDY(px, py, sx, sy, vx, vy) {
    const dx = px - sx, dy = py - sy;
    const norm2 = dx * dx + dy * dy;
    const k     = Math.exp(-norm2 / GAMMA);
    const sxv = -px, syv = -py;
    const dDotv = dx * vx + dy * vy;
    const vDots = vx * sxv + vy * syv;
    const dDots = dx * sxv + dy * syv;
    const Cd = (2 / GAMMA) * dDotv + vDots;
    const Cv = (D - 1) - (2 / GAMMA) * norm2 - dDots;
    const f  = (2 / GAMMA) * k;
    return [f * (Cd * dx + Cv * vx), f * (Cd * dy + Cv * vy)];
  }

  // -------- RK4 for the particle ODE  dx/dt = w_g * psi(x; src, v) --------
  function rk4Step(px, py, dt, sx, sy, vx, vy, fieldFn) {
    const h2 = dt * 0.5;
    const f1 = fieldFn(px, py, sx, sy, vx, vy);
    const k1x = W_G * f1[0], k1y = W_G * f1[1];
    const f2 = fieldFn(px + h2 * k1x, py + h2 * k1y, sx, sy, vx, vy);
    const k2x = W_G * f2[0], k2y = W_G * f2[1];
    const f3 = fieldFn(px + h2 * k2x, py + h2 * k2y, sx, sy, vx, vy);
    const k3x = W_G * f3[0], k3y = W_G * f3[1];
    const f4 = fieldFn(px + dt * k3x, py + dt * k3y, sx, sy, vx, vy);
    const k4x = W_G * f4[0], k4y = W_G * f4[1];
    const sixth = dt / 6;
    return [
      px + sixth * (k1x + 2 * k2x + 2 * k3x + k4x),
      py + sixth * (k1y + 2 * k2y + 2 * k3y + k4y),
    ];
  }

  // ============================================================
  //  Per-panel particle simulation
  // ============================================================
  class Panel {
    constructor(canvasId, fieldFn, kind) {
      this.canvas  = document.getElementById(canvasId);
      this.ctx     = this.canvas.getContext("2d");
      this.fieldFn = fieldFn;
      this.kind    = kind;             // "EDDY" or "PG"

      this.particles = [];             // array of [x, y]
      this.ages      = [];             // parallel: age in frames

      this.srcX = 0; this.srcY = 0;
      this.vX   = 1; this.vY   = 0;

      this.resize();
    }

    resize() {
      const dpr  = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      this.canvas.width  = Math.max(1, Math.round(rect.width  * dpr));
      this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.cssW = rect.width;
      this.cssH = rect.height;
    }

    w2sX(x) { return ((x + VIEW_HALF) / (2 * VIEW_HALF)) * this.cssW; }
    w2sY(y) { return (1 - (y + VIEW_HALF) / (2 * VIEW_HALF)) * this.cssH; }

    // -------- Episode setup --------
    setState(srcPos, srcV) {
      this.srcX = srcPos[0];
      this.srcY = srcPos[1];
      this.vX   = srcV[0];
      this.vY   = srcV[1];
      this.initParticles();
    }

    initParticles() {
      this.particles = new Array(N_PARTICLES);
      this.ages      = new Array(N_PARTICLES);
      const tmp = [0, 0];
      const sx = this.srcX, sy = this.srcY;
      const vx = this.vX,   vy = this.vY;
      const ff = this.fieldFn;
      for (let i = 0; i < N_PARTICLES; i++) {
        sampleFromP(this.kind, sx, sy, tmp);
        let x = tmp[0], y = tmp[1];
        // Stagger ages uniformly in [0, LIFESPAN) so deaths are uniform
        // in time. To keep position-vs-age consistent (i.e. the cloud is
        // already at the t=∞ steady state — no warmup), evolve each
        // particle forward by `age` integration steps under psi.
        const age = Math.floor(Math.random() * LIFESPAN_FRAMES);
        for (let s = 0; s < age; s++) {
          const next = rk4Step(x, y, SIM_DT, sx, sy, vx, vy, ff);
          x = next[0]; y = next[1];
        }
        this.particles[i] = [x, y];
        this.ages[i] = age;
      }
    }

    // -------- Time step --------
    step(dt) {
      const sx = this.srcX, sy = this.srcY;
      const vx = this.vX,   vy = this.vY;
      const ff = this.fieldFn;
      const ps = this.particles, ages = this.ages;
      const tmp = [0, 0];
      for (let i = 0; i < ps.length; i++) {
        const p = ps[i];
        if (++ages[i] >= LIFESPAN_FRAMES) {
          // Respawn: sample from the analytical p_t.
          sampleFromP(this.kind, sx, sy, tmp);
          p[0] = tmp[0]; p[1] = tmp[1];
          ages[i] = 0;
          continue;
        }
        // Pure transport: dx/dt = w_g * psi.  RK4. No OU, no noise.
        const next = rk4Step(p[0], p[1], dt, sx, sy, vx, vy, ff);
        p[0] = next[0];
        p[1] = next[1];
      }
    }

    // -------- Drawing --------
    draw() {
      const ctx = this.ctx;
      ctx.fillStyle = "#fbfbfd";
      ctx.fillRect(0, 0, this.cssW, this.cssH);

      this.drawContours();
      this.drawParticles();
      this.drawSource();
    }

    drawContours() {
      const ctx = this.ctx;
      const cx = this.w2sX(0), cy = this.w2sY(0);
      const pxPerUnit = this.cssW / (2 * VIEW_HALF);
      // Darker, slightly thicker dashes so the N(0, I_2) reference is easy to read.
      ctx.strokeStyle = "rgba(20, 22, 30, 0.75)";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.25;
      for (const r of [1, 2, 3]) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * pxPerUnit, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    drawParticles() {
      // Each disk is drawn in its own fill() so per-particle alpha
      // accumulates through overlap (smooth density blob). Alpha follows
      // a sin life envelope so newborn / dying particles fade in/out.
      const ctx = this.ctx;
      const ps   = this.particles;
      const ages = this.ages;
      const r = DOT_RADIUS;
      const W = this.cssW, H = this.cssH;
      const margin = r + 2;
      const invLife = 1 / LIFESPAN_FRAMES;
      for (let i = 0; i < ps.length; i++) {
        const p = ps[i];
        const sx = this.w2sX(p[0]);
        const sy = this.w2sY(p[1]);
        if (sx < -margin || sx > W + margin || sy < -margin || sy > H + margin) continue;
        const lifeFade = Math.sin(Math.PI * ages[i] * invLife);
        if (lifeFade < 0.02) continue;
        const a = PARTICLE_BASE_A * lifeFade;
        ctx.fillStyle = `rgba(0, 0, 0, ${a})`;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    drawSource() {
      const sx = this.w2sX(this.srcX);
      const sy = this.w2sY(this.srcY);
      // The v-arrow is meaningful only for EDDY-RBF — PG ignores v entirely.
      // Draw it before the star so the star sits on top of the arrow's tail.
      if (this.kind === "EDDY") {
        // World y is flipped vs screen y, so map (vX, vY) -> screen direction.
        this.drawArrow(sx, sy, this.vX, -this.vY);
      }
      this.drawStar(sx, sy, STAR_RADIUS);
    }

    drawStar(cx, cy, r) {
      const ctx = this.ctx;
      const points = 5;
      const inner  = r * 0.45;
      ctx.beginPath();
      for (let k = 0; k < points * 2; k++) {
        const radius = (k % 2 === 0) ? r : inner;
        const angle  = (k * Math.PI) / points - Math.PI / 2;
        const px = cx + radius * Math.cos(angle);
        const py = cy + radius * Math.sin(angle);
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = "#f5b400";
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = "rgba(60, 40, 0, 0.7)";
      ctx.stroke();
    }

    // Draws a gold/dark-stroke arrow from a star at (cx, cy) in screen coords,
    // pointing in screen direction (sdx, sdy). Same fill/stroke as the star.
    drawArrow(cx, cy, sdx, sdy) {
      const norm = Math.hypot(sdx, sdy) || 1;
      const ux = sdx / norm, uy = sdy / norm;     // unit forward (screen)
      const px = -uy,        py = ux;              // unit perpendicular (screen)

      // Geometry in pixels along (ux, uy):
      const offset    = STAR_RADIUS + 2;           // start just outside the star
      const shaftLen  = 18;                         // shaft length
      const headLen   = 12;                         // arrowhead length
      const shaftHalf = 2.5;                        // half thickness of shaft
      const headHalf  = 7.5;                        // half width of arrowhead
      const total     = shaftLen + headLen;

      // 7-point closed polygon: rectangular shaft + triangular head.
      const pts = [
        [offset,                 -shaftHalf],
        [offset + shaftLen,      -shaftHalf],
        [offset + shaftLen,      -headHalf],
        [offset + total,          0],
        [offset + shaftLen,       headHalf],
        [offset + shaftLen,       shaftHalf],
        [offset,                  shaftHalf],
      ];

      const ctx = this.ctx;
      ctx.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i][0], b = pts[i][1];
        const x = cx + a * ux + b * px;
        const y = cy + a * uy + b * py;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = "#f5b400";
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = "rgba(60, 40, 0, 0.7)";
      ctx.stroke();
    }
  }

  // ============================================================
  //  Top-level orchestration
  //
  //  Source state is fixed for both panels (no episode timer, no
  //  user controls): the star sits at the origin, v points right.
  //  initParticles() advances each particle through its initial age,
  //  so the first frame is already at the t=∞ steady state.
  // ============================================================
  const FIXED_SRC = [0, 0];
  const FIXED_V   = [1, 0];
  let pgPanel, eddyPanel;

  function frame() {
    for (let s = 0; s < SUB_STEPS; s++) {
      pgPanel.step(SIM_DT);
      eddyPanel.step(SIM_DT);
    }
    pgPanel.draw();
    eddyPanel.draw();
    requestAnimationFrame(frame);
  }

  document.addEventListener("DOMContentLoaded", () => {
    pgPanel   = new Panel("canvas-pg",   fieldPG,   "PG");
    eddyPanel = new Panel("canvas-eddy", fieldEDDY, "EDDY");

    pgPanel.setState(FIXED_SRC, FIXED_V);
    eddyPanel.setState(FIXED_SRC, FIXED_V);

    requestAnimationFrame(frame);

    let resizeT = 0;
    window.addEventListener("resize", () => {
      clearTimeout(resizeT);
      resizeT = setTimeout(() => {
        pgPanel.resize();
        eddyPanel.resize();
      }, 100);
    });
  });
})();

/* ================================================================
 *  Section 2 — VP-DDPM: EDDY on a 5-mode Gaussian mixture
 *
 *  Target:  p_0 = (1/L) Σ_l N(K·u_l, I),  u_l = [cos(2πl/L), sin(2πl/L)]
 *  Forward (VP-DDPM):  dx = -½β(t)x dt + √β(t) dW
 *    β(t)   = BMIN + (BMAX-BMIN)·t/T
 *    α(t)   = exp(-½∫₀ᵗ β)           (scale factor)
 *    p_t(x) = (1/L) Σ_l N(α(t)·K·u_l, I)   (variance always 1)
 *  Score:   s(x,t) = Σ_l w_l(x,t)·(α(t)K·u_l - x)   (softmax weights)
 *  Reverse probability flow ODE (t: T → 0):
 *    dx = [½β(t)x + ½β(t)·s(x,t)] dt
 *  EDDY correction (eq. 40, 2D analytic anti-symmetric A_i):
 *    A_i = [[0, aᵢ], [-aᵢ, 0]],  aᵢ = (-2/γ(n-1)) Σ_{j≠i} k_ij·cross(v_i, δ_ij)
 *    EDDY_i = A_i·s_i + div(A_i)
 *           = [aᵢ·sᵧ + ∂aᵢ/∂y,  -aᵢ·sₓ - ∂aᵢ/∂x]
 * ================================================================ */
(() => {
  const BMIN = 0.1, BMAX = 20.0, T_END = 1.0;
  const K_RAD = 5.0, N_MOD = 5;         // mode radius, mode count
  const WG = -3.5, GM = 1.0;            // EDDY guidance weight, RBF bandwidth
  // T_START≈0.7: α(0.7)≈0.084 so p_{T_START}≈N(0,I) and K·(1−α)≈4.6 units of travel
  const T_START = 0.7;
  const ITERS = 700;                     // ODE steps per trajectory
  const DT = T_START / ITERS;
  const BATCH_SZ    = N_MOD;            // 5 particles per batch (one per mode)
  const STEPS_FRAME = 4;                // ODE steps per animation frame
  const VH2 = 8.0;                       // world half-extent (modes at radius 5)
  const TRAIL_MAX   = 10000;
  const HOLD_FRAMES  = 60;             // pause after batch completes before reset
  // Background particle cloud — fixed N(0, BG_SIG²·I), same style as section 1
  const N_BG      = 5_000;
  const BG_LIFE   = 60;
  const BG_SIG    = 3.5;
  const BG_DOT_R  = 3;
  const BG_ALPHA  = 0.1;

  // Five curated starting-point sets where EDDY covers all 5 modes, IID covers 2.
  // Selected by scanning mulberry32 seeds 1–50000 under WG=-3.5, GM=1.0.
  const REPEAT_PTS = [
    [[-0.7604, 0.5676], [-0.5815, -1.894 ], [-0.2364, -0.7644], [ 0.4966, -0.5911], [-0.601,   0.2809]],
    [[ 0.5239, 0.4875], [ 0.2036,  2.0946], [ 0.238,  -1.0312], [ 0.0585, -1.6112], [-0.246,   0.9842]],
    [[ 0.7054, 0.642 ], [ 0.24,    0.3151], [-0.3253,  1.2442], [-0.1087, -1.1887], [ 0.4534, -0.6634]],
    [[ 0.5616, 0.1988], [ 0.8267, -0.5656], [-0.6427,  0.8943], [-1.9026,  0.1641], [-0.8471,  0.7014]],
    [[ 0.9105, 0.5069], [ 1.0072,  0.167 ], [-1.3623,  0.9111], [ 1.2468, -0.278 ], [-1.4429,  0.0132]],
  ];
  let repeatIdx = 0;

  const PCOLS2 = ["#e74c3c","#3498db","#27ae60","#f39c12","#9b59b6"];
  const PCOLS2_RGB = PCOLS2.map(h => [
    parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)
  ]);

  const rn2 = () => {
    const u = Math.random() || 1e-12;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  };

  // --- Marginal: p_t(x) = (1/L) Σ_l N(x; α_t K u_l, I)
  //     Total variance is always 1: α_t²·σ_data² + σ_t² = α_t² + (1-α_t²) = 1
  function vpAlpha(t) {
    return Math.exp(-0.5 * (BMIN * t + (BMAX - BMIN) * t * t / (2 * T_END)));
  }

  // --- Score (log-sum-exp stable, variance = 1) ---
  function vpScore(x, y, t) {
    const a = vpAlpha(t);
    let maxLW = -Infinity;
    const lws = new Float64Array(N_MOD);
    for (let l = 0; l < N_MOD; l++) {
      const th = 2 * Math.PI * l / N_MOD;
      const mx = a * K_RAD * Math.cos(th), my = a * K_RAD * Math.sin(th);
      lws[l] = -0.5 * ((x - mx) ** 2 + (y - my) ** 2);  // variance = 1
      if (lws[l] > maxLW) maxLW = lws[l];
    }
    let ws = 0, rx = 0, ry = 0;
    for (let l = 0; l < N_MOD; l++) {
      const th = 2 * Math.PI * l / N_MOD;
      const mx = a * K_RAD * Math.cos(th), my = a * K_RAD * Math.sin(th);
      const w = Math.exp(lws[l] - maxLW);
      ws += w; rx += w * (mx - x); ry += w * (my - y);  // divide by 1
    }
    return [rx / ws, ry / ws];
  }

  // --- EDDY guidance: analytic 2D anti-symmetric A_i @ score + div(A_i) ---
  function eddyGuide(pts, vels, scs) {
    const n = pts.length;
    return pts.map((_, i) => {
      const [xi, yi] = pts[i], [vx, vy] = vels[i];
      let ai = 0, dadx = 0, dady = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = xi - pts[j][0], dy = yi - pts[j][1];
        const k  = Math.exp(-(dx * dx + dy * dy) / GM);
        const cr = vx * dy - vy * dx;
        ai   += k * cr;
        dadx += k * ((-2 * dx / GM) * cr - vy);
        dady += k * ((-2 * dy / GM) * cr + vx);
      }
      const c = -2 / (GM * (n - 1));
      ai *= c; dadx *= c; dady *= c;
      const [sx, sy] = scs[i];
      return [ai * sy + dady, -ai * sx - dadx];
    });
  }

  // Probability flow ODE step: dx = [½β(t)x + ½β(t)·s(x,t)] dt  (no noise)
  function eddyBatchStep(pts, t) {
    const beta = BMIN + (BMAX - BMIN) * t / T_END;
    const scs  = pts.map(([x, y]) => vpScore(x, y, t));
    const vels = pts.map(([x, y], i) => {
      const [sx, sy] = scs[i];
      return [0.5 * beta * x + 0.5 * beta * sx, 0.5 * beta * y + 0.5 * beta * sy];
    });
    const eg = eddyGuide(pts, vels, scs);
    return pts.map(([x, y], i) => {
      const [vx, vy] = vels[i], [ex, ey] = eg[i];
      return [
        x + (vx + WG * ex) * DT,
        y + (vy + WG * ey) * DT,
      ];
    });
  }

  // IID step: pure probability flow ODE (no guidance)
  function iidBatchStep(pts, t) {
    const beta = BMIN + (BMAX - BMIN) * t / T_END;
    return pts.map(([x, y]) => {
      const [sx, sy] = vpScore(x, y, t);
      return [
        x + (0.5 * beta * x + 0.5 * beta * sx) * DT,
        y + (0.5 * beta * y + 0.5 * beta * sy) * DT,
      ];
    });
  }

  function generateSpreadPts() {
    const pts = REPEAT_PTS[repeatIdx].map(p => [...p]);
    repeatIdx = (repeatIdx + 1) % REPEAT_PTS.length;
    return pts;
  }

  // ================================================================
  //  VpBatch — BATCH_SZ particles marching from t = T_START down to 0
  // ================================================================
  class VpBatch {
    constructor(useEddy = true) {
      this.useEddy = useEddy;
      this.step    = 0;
      this.pts     = [];
      this.trails  = [];
    }

    start(pts) {
      this.pts    = pts.map(p => [...p]);
      this.step   = 0;
      this.trails = Array.from({ length: BATCH_SZ }, () => []);
    }

    _tick() {
      const t = T_START - this.step * DT;
      this.pts = this.useEddy ? eddyBatchStep(this.pts, t) : iidBatchStep(this.pts, t);
      this.step++;
    }

    advance() {
      if (this.step >= ITERS) return;
      this._tick();
      for (let i = 0; i < BATCH_SZ; i++) {
        this.trails[i].push([...this.pts[i]]);
        if (this.trails[i].length > TRAIL_MAX) this.trails[i].shift();
      }
    }

    done()   { return this.step >= ITERS; }
    finals() { return this.pts.map(p => [...p]); }
  }

  // ================================================================
  //  VpPanel — canvas for one run (EDDY or IID)
  // ================================================================
  class VpPanel {
    constructor(id, useEddy = true) {
      this.canvas  = document.getElementById(id);
      if (!this.canvas) return;
      this.ctx     = this.canvas.getContext("2d");
      this.useEddy = useEddy;
      this.batch   = new VpBatch(useEddy);
      this.lastVels = Array.from({ length: BATCH_SZ }, () => [0, 0]);
      if (useEddy) {
        this.bgPts         = Array.from({ length: N_BG }, () => [rn2() * BG_SIG, rn2() * BG_SIG]);
        this.bgAges        = Array.from({ length: N_BG }, () => (Math.random() * BG_LIFE) | 0);
        this.bgForcePerSrc = new Float32Array(N_BG * BATCH_SZ);
      }
      this.resize();
    }

    startEpisode(pts) {
      this.batch.start(pts);
      if (this.useEddy) {
        this.bgPts   = Array.from({ length: N_BG }, () => [rn2() * BG_SIG, rn2() * BG_SIG]);
        this.bgAges  = Array.from({ length: N_BG }, () => (Math.random() * BG_LIFE) | 0);
        this.bgForcePerSrc.fill(0);
      }
    }

    resize() {
      if (!this.canvas) return;
      const dpr  = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      this.canvas.width  = Math.max(1, Math.round(rect.width  * dpr));
      this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.cssW = rect.width;
      this.cssH = rect.height;
    }

    sx(x) { return ((x + VH2) / (2 * VH2)) * this.cssW; }
    sy(y) { return (1 - (y + VH2) / (2 * VH2)) * this.cssH; }

    tick() {
      if (!this.canvas || !this.useEddy) return;
      // Move background particles under the EDDY ψ field from the colored particles
      const effDT  = DT * STEPS_FRAME;
      const curT   = Math.max(T_START - this.batch.step * DT, 0.03);
      const beta   = BMIN + (BMAX - BMIN) * curT / T_END;
      const cPts   = this.batch.pts;
      const cVels  = cPts.map(([cx, cy]) => {
        const [csx, csy] = vpScore(cx, cy, curT);
        return [0.5 * beta * cx + 0.5 * beta * csx,
                0.5 * beta * cy + 0.5 * beta * csy];
      });
      this.lastVels = cVels;
      for (let i = 0; i < N_BG; i++) {
        if (++this.bgAges[i] >= BG_LIFE) {
          this.bgPts[i] = [rn2() * BG_SIG, rn2() * BG_SIG];
          this.bgAges[i] = 0;
          this.bgForcePerSrc.fill(0, i * BATCH_SZ, i * BATCH_SZ + BATCH_SZ);
          continue;
        }
        let [bx, by] = this.bgPts[i];
        const [bsx, bsy] = vpScore(bx, by, curT);
        // Accumulate combined displacement and per-source max forces
        let ex = 0, ey = 0;
        const base = i * BATCH_SZ;
        for (let j = 0; j < BATCH_SZ; j++) {
          const [vx, vy] = cVels[j];
          const dx = bx - cPts[j][0], dy = by - cPts[j][1];
          const k  = Math.exp(-(dx * dx + dy * dy) / GM);
          const cr = vx * dy - vy * dx;
          const c  = -2 / GM;
          const ai   = c * k * cr;
          const dadx = c * k * ((-2 * dx / GM) * cr - vy);
          const dady = c * k * ((-2 * dy / GM) * cr + vx);
          const fex = ai * bsy + dady, fey = -ai * bsx - dadx;
          ex += fex; ey += fey;
          const fMag = Math.abs(WG * effDT) * Math.sqrt(fex * fex + fey * fey);
          if (fMag > this.bgForcePerSrc[base + j]) this.bgForcePerSrc[base + j] = fMag;
        }
        let dx = WG * ex * effDT, dy = WG * ey * effDT;
        const dMag = Math.sqrt(dx * dx + dy * dy);
        if (dMag > 0.4) { dx *= 0.4 / dMag; dy *= 0.4 / dMag; }
        this.bgPts[i] = [bx + dx, by + dy];
      }
    }

    draw() {
      if (!this.canvas) return;
      const ctx = this.ctx;
      ctx.fillStyle = "#fbfbfd";
      ctx.fillRect(0, 0, this.cssW, this.cssH);

      // Background particle cloud — EDDY panel only
      if (this.useEddy && this.bgPts) {
        const invBgLife = 1 / BG_LIFE;
        for (let i = 0; i < N_BG; i++) {
          const base = i * BATCH_SZ;
          let totalForce = 0;
          for (let j = 0; j < BATCH_SZ; j++) totalForce += this.bgForcePerSrc[base + j];
          const forceAlpha = Math.min(totalForce / 0.05, 1.0);
          if (forceAlpha < 0.01) continue;
          const fade = Math.sin(Math.PI * this.bgAges[i] * invBgLife);
          if (fade < 0.02) continue;
          let cr = 0, cg = 0, cb = 0;
          for (let j = 0; j < BATCH_SZ; j++) {
            const w = this.bgForcePerSrc[base + j] / totalForce;
            cr += w * PCOLS2_RGB[j][0]; cg += w * PCOLS2_RGB[j][1]; cb += w * PCOLS2_RGB[j][2];
          }
          ctx.globalAlpha = BG_ALPHA * fade * forceAlpha;
          ctx.fillStyle   = `rgb(${cr|0},${cg|0},${cb|0})`;
          ctx.beginPath();
          ctx.arc(this.sx(this.bgPts[i][0]), this.sy(this.bgPts[i][1]), BG_DOT_R, 0, 2 * Math.PI);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // 5 mode markers — outer 2σ ring only
      const pu = this.cssW / (2 * VH2);
      ctx.save();
      ctx.strokeStyle = "#000";
      ctx.lineWidth   = 1.25;
      ctx.setLineDash([4, 3]);
      ctx.globalAlpha = 0.75;
      for (let l = 0; l < N_MOD; l++) {
        const th = 2 * Math.PI * l / N_MOD;
        ctx.beginPath();
        ctx.arc(this.sx(K_RAD * Math.cos(th)), this.sy(K_RAD * Math.sin(th)), 2 * pu, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.restore();

      // Trajectory trails and particle dots
      const b = this.batch;
      for (let i = 0; i < BATCH_SZ; i++) {
        const trail = b.trails[i], col = PCOLS2[i];
        if (trail.length >= 2) {
          for (let k = 1; k < trail.length; k++) {
            ctx.globalAlpha = 1.00;
            ctx.strokeStyle = col;
            ctx.lineWidth   = 2.5;
            ctx.beginPath();
            ctx.moveTo(this.sx(trail[k-1][0]), this.sy(trail[k-1][1]));
            ctx.lineTo(this.sx(trail[k][0]),   this.sy(trail[k][1]));
            ctx.stroke();
          }
        }
        if (b.pts[i]) {
          ctx.globalAlpha = 1.0;
          this._drawStar(ctx, this.sx(b.pts[i][0]), this.sy(b.pts[i][1]), 10, col);
        }
      }
      ctx.globalAlpha = 1;
    }

    _drawStar(ctx, cx, cy, r, fill) {
      const inner = r * 0.45;
      ctx.beginPath();
      for (let k = 0; k < 10; k++) {
        const radius = (k % 2 === 0) ? r : inner;
        const angle  = (k * Math.PI) / 5 - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth   = 1.2;
      ctx.strokeStyle = "rgba(0,0,0,0.45)";
      ctx.stroke();
    }

  }

  // --- Bootstrap ---
  let eddyPanel2, iidPanel2;
  let episodeHold = 0;

  function startEpisode() {
    episodeHold = 0;
    const pts = generateSpreadPts();
    eddyPanel2.startEpisode(pts);
    iidPanel2.startEpisode(pts);
  }

  function vpLoop() {
    if (eddyPanel2.batch.done()) {
      if (++episodeHold >= HOLD_FRAMES) startEpisode();
    } else {
      for (let s = 0; s < STEPS_FRAME; s++) {
        eddyPanel2.batch.advance();
        iidPanel2.batch.advance();
      }
    }
    eddyPanel2.tick();
    eddyPanel2.draw();
    iidPanel2.draw();
    requestAnimationFrame(vpLoop);
  }

  document.addEventListener("DOMContentLoaded", () => {
    eddyPanel2 = new VpPanel("canvas-vp-eddy", true);
    iidPanel2  = new VpPanel("canvas-vp-iid",  false);
    if (eddyPanel2.canvas && iidPanel2.canvas) {
      startEpisode();
      requestAnimationFrame(vpLoop);
    }

    let _vpResT;
    window.addEventListener("resize", () => {
      clearTimeout(_vpResT);
      _vpResT = setTimeout(() => {
        eddyPanel2.resize();
        iidPanel2.resize();
      }, 100);
    });
  });
})();
