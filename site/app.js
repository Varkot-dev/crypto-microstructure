/**
 * Binance microstructure results site.
 *
 * Loads the four derived slices written by build_data.py and draws them.
 * Every displayed number originates in results/*.json; nothing is computed
 * here beyond formatting, axis ranges, and the OLS lines whose coefficients
 * come from the source artifacts' own regression blocks.
 */

const DATA = {
  crossSection: 'data/cross_section.json',
  kernels: 'data/kernels.json',
  endogeneity: 'data/endogeneity.json',
  execution: 'data/execution.json',
};

const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
};

/* ---------- theme plumbing -------------------------------------------- */

const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function theme() {
  return {
    ink: css('--ink'),
    inkSoft: css('--ink-soft'),
    inkFaint: css('--ink-faint'),
    rule: css('--rule'),
    plot: css('--plot'),
    plot2: css('--plot-2'),
    band: css('--band'),
    flag: css('--flag'),
    paper: css('--paper-sunk'),
    mono: css('--face-data') || 'monospace',
  };
}

function baseLayout(t, overrides = {}) {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: t.mono, size: 11, color: t.inkFaint },
    margin: { l: 56, r: 18, t: 14, b: 46 },
    hoverlabel: {
      bgcolor: t.paper,
      bordercolor: t.rule,
      font: { family: t.mono, size: 11, color: t.ink },
      align: 'left',
    },
    showlegend: false,
    ...overrides,
  };
}

function axis(t, title, extra = {}) {
  return {
    title: { text: title, font: { size: 11, color: t.inkFaint } },
    gridcolor: t.rule,
    zerolinecolor: t.rule,
    linecolor: t.rule,
    tickfont: { size: 10, color: t.inkFaint },
    automargin: true,
    ...extra,
  };
}

/* ---------- formatting ------------------------------------------------ */

const fmt = (x, d = 4) => Number(x).toFixed(d);
const int = (x) => Number(x).toLocaleString('en-US');
const sign = (x, d = 4) => (x >= 0 ? '+' : '') + Number(x).toFixed(d);

/** Interpolate a colour ramp by activity rank — low to high. */
function activityColor(frac, t) {
  // Blend the two measurement hues: quiet symbols cool, active symbols
  // saturated. Colour here encodes activity, which is the panel's
  // organising variable, not decoration.
  const a = hexish(t.plot2);
  const b = hexish(t.plot);
  if (!a || !b) return t.plot;
  const mix = a.map((v, i) => Math.round(v + (b[i] - v) * frac));
  return `rgb(${mix.join(',')})`;
}

/** Resolve any computed colour (oklch included) to [r,g,b] via canvas. */
const _resolveCache = new Map();
function hexish(color) {
  if (_resolveCache.has(color)) return _resolveCache.get(color);
  try {
    const c = document.createElement('canvas').getContext('2d');
    c.fillStyle = '#000';
    c.fillStyle = color;
    c.fillRect(0, 0, 1, 1);
    const d = c.getImageData(0, 0, 1, 1).data;
    const out = [d[0], d[1], d[2]];
    _resolveCache.set(color, out);
    return out;
  } catch {
    return null;
  }
}

/* ---------- loading --------------------------------------------------- */

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

function failPlot(id, err) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '';
  const p = document.createElement('p');
  p.className = 'load-error';
  p.textContent = `Could not load this chart: ${err.message}`;
  el.appendChild(p);
  console.error(`[${id}]`, err);
}

/* ============================================================
   1 — Cross-section scatter
   ============================================================ */

const CS_AXES = [
  {
    key: 'p_flip',
    label: 'p_flip',
    title: 'P(next trade flips sign)',
    reference: 0.5,
    referenceLabel: 'p_flip = 0.5 · no persistence',
    regression: 'p_flip_vs_activity',
    digits: 4,
  },
  {
    key: 'gamma',
    label: 'γ',
    title: 'Sign-ACF exponent γ̂',
    regression: 'gamma_vs_activity',
    digits: 4,
  },
  { key: 'acf1', label: 'ACF(1)', title: 'Lag-1 sign autocorrelation', digits: 4 },
  {
    key: 'zigzag_amplitude',
    label: 'zigzag',
    title: 'Zigzag amplitude (even − odd lags)',
    digits: 4,
  },
];

function drawCrossSection(data, axisSpec) {
  const t = theme();
  const el = document.getElementById('cs-plot');
  const panel = data.symbols.filter((s) => s.in_kernel_panel);
  const rest = data.symbols.filter((s) => !s.in_kernel_panel);

  const hover = (s) =>
    [
      `<b>${s.symbol}</b>`,
      `n_events   ${int(s.n_events)}`,
      `γ̂          ${fmt(s.gamma)} ± ${fmt(s.stderr)}`,
      `ACF(1)     ${fmt(s.acf1)}`,
      `p_flip     ${fmt(s.p_flip)}`,
      `zigzag     ${fmt(s.zigzag_amplitude)}`,
    ].join('<br>');

  const point = (arr, color, size, name, symbol = 'circle') => ({
    type: 'scatter',
    mode: 'markers',
    name,
    x: arr.map((s) => s.log_n),
    y: arr.map((s) => s[axisSpec.key]),
    text: arr.map(hover),
    hovertemplate: '%{text}<extra></extra>',
    marker: {
      color,
      size,
      symbol,
      opacity: 0.85,
      line: { width: symbol === 'circle' ? 0 : 1.5, color },
    },
  });

  const traces = [
    point(rest, t.plot, 7, 'symbol'),
    point(panel, t.flag, 10, 'kernel panel', 'circle-open'),
  ];

  const xs = data.symbols.map((s) => s.log_n);
  const xMin = Math.min(...xs) - 0.05;
  const xMax = Math.max(...xs) + 0.05;

  // OLS line drawn from the artifact's own regression block — never refit here.
  const reg = axisSpec.regression ? data.regressions[axisSpec.regression] : null;
  if (reg) {
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: [xMin, xMax],
      y: [reg.intercept + reg.slope * xMin, reg.intercept + reg.slope * xMax],
      line: { color: t.ink, width: 1.5, dash: 'solid' },
      hovertemplate:
        `OLS  slope ${fmt(reg.slope)} ± ${fmt(reg.stderr)}` +
        `<br>R² ${fmt(reg.r2, 4)} · n ${reg.n}<extra></extra>`,
    });
  }

  const shapes = [];
  if (axisSpec.reference !== undefined) {
    shapes.push({
      type: 'line',
      xref: 'paper',
      x0: 0,
      x1: 1,
      y0: axisSpec.reference,
      y1: axisSpec.reference,
      line: { color: t.band, width: 1.5, dash: 'dash' },
      layer: 'below',
    });
  }

  const layout = baseLayout(t, {
    xaxis: axis(t, 'log₁₀ aggressor events', { range: [xMin, xMax] }),
    yaxis: axis(t, axisSpec.title),
    shapes,
  });

  Plotly.react(el, traces, layout, PLOT_CONFIG);

  const note = document.getElementById('cs-note');
  const parts = [
    `${data.n_successful} symbols of ${data.n_requested} requested; ` +
      `${data.n_skipped} fell below the ${int(data.min_events)}-event floor.`,
  ];
  if (reg) {
    parts.push(
      `OLS slope ${fmt(reg.slope)} (stderr ${fmt(reg.stderr)}), ` +
        `intercept ${fmt(reg.intercept)}, R² ${fmt(reg.r2, 4)}, n = ${reg.n}.`,
    );
  } else {
    parts.push('No fitted line: this statistic has no regression in the source artifact.');
  }
  if (axisSpec.referenceLabel) parts.push(`Dashed line: ${axisSpec.referenceLabel}.`);
  note.textContent = parts.join(' ');
}

function initCrossSection(data) {
  const controls = document.getElementById('cs-controls');
  let active = CS_AXES[0];

  CS_AXES.forEach((spec) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = spec.label;
    b.setAttribute('aria-pressed', String(spec === active));
    b.addEventListener('click', () => {
      active = spec;
      [...controls.children].forEach((c) =>
        c.setAttribute('aria-pressed', String(c === b)),
      );
      drawCrossSection(data, active);
    });
    controls.appendChild(b);
  });

  const t = theme();
  document.getElementById('cs-legend').innerHTML = `
    <span><i style="background:${t.plot}"></i>121-symbol cross-section</span>
    <span><i style="border:2px solid ${t.flag}"></i>also in the kernel panel</span>`;

  drawCrossSection(data, active);

  const tails = data.tails;
  const p = document.createElement('p');
  p.className = 'prose-note';
  p.style.marginTop = '1rem';
  p.textContent =
    `Tails, on p_flip: ${tails.most_active_anti_persistent} of the ` +
    `${tails.tail_size} most-active symbols flip more often than a fair coin ` +
    `(p_flip > 0.5), against ${tails.least_active_anti_persistent} of the ` +
    `${tails.tail_size} least-active.`;
  document.getElementById('cs-plot').closest('.plot-frame').after(p);
}

/* ============================================================
   2 — Kernel panel
   ============================================================ */

function initKernels(data) {
  const t = theme();
  const el = document.getElementById('k-plot');
  const recs = data.records;
  const n = recs.length;

  const traces = recs.map((r, i) => {
    // Rank by activity: records arrive sorted descending, so invert.
    const frac = n > 1 ? (n - 1 - i) / (n - 1) : 1;
    const lags = [];
    const vals = [];
    r.G.forEach((g, lag) => {
      if (lag >= 1 && g > 0) {
        lags.push(lag);
        vals.push(g);
      }
    });
    return {
      type: 'scatter',
      mode: 'lines',
      name: r.symbol,
      x: lags,
      y: vals,
      line: { color: activityColor(frac, t), width: 1.6 },
      opacity: r.verdict === 'violated' ? 0.55 : 0.95,
      hovertemplate:
        `<b>${r.symbol}</b><br>` +
        `lag %{x} · G %{y:.3e}<br>` +
        `γ̂_week ${fmt(r.gamma_week)} · β̂ ${fmt(r.beta)}<br>` +
        `Δ ${sign(r.balance_delta)} · ${r.verdict}<extra></extra>`,
    };
  });

  const layout = baseLayout(t, {
    xaxis: axis(t, 'lag ℓ (events)', { type: 'log' }),
    yaxis: axis(t, 'G(ℓ)', { type: 'log' }),
  });

  Plotly.react(el, traces, layout, PLOT_CONFIG);

  document.getElementById('k-legend').innerHTML = `
    <span><i style="background:${activityColor(0, t)}"></i>least active</span>
    <span><i style="background:${activityColor(1, t)}"></i>most active</span>`;

  document.getElementById('k-note').textContent =
    `${data.n_successful} symbols, ${data.start_day} to ${data.end_day}, max lag ` +
    `${data.max_lag}. Only positive G values are plotted, since the axes are ` +
    `logarithmic. Curves are the bare kernel recovered by deconvolution, not the ` +
    `measured response.`;

  drawBalanceStrip(data, t);
}

function drawBalanceStrip(data, t) {
  const strip = document.getElementById('k-strip');
  // Symmetric domain wide enough to hold every delta and every band.
  const extent = Math.max(
    ...data.records.map((r) => Math.abs(r.balance_delta) + 0.01),
    ...data.records.map((r) => r.band + 0.01),
  );
  const pct = (v) => ((v + extent) / (2 * extent)) * 100;

  data.records.forEach((r) => {
    const row = document.createElement('div');
    row.className = 'strip-row';
    row.dataset.verdict = r.verdict;

    const bandLeft = pct(-r.band);
    const bandWidth = pct(r.band) - bandLeft;

    row.innerHTML = `
      <span class="strip-sym">${r.symbol}</span>
      <span class="strip-track" role="img"
            aria-label="${r.symbol}: balance delta ${sign(r.balance_delta)}, band plus or minus ${fmt(r.band, 4)}, ${r.verdict}">
        <span class="strip-band" style="left:${bandLeft}%;width:${bandWidth}%"></span>
        <span class="strip-zero" style="left:${pct(0)}%"></span>
        <span class="strip-delta" style="left:${pct(r.balance_delta)}%"></span>
      </span>
      <span class="strip-val">${sign(r.balance_delta)}</span>`;

    row.title =
      `${r.symbol} · γ̂_week ${fmt(r.gamma_week)} · β̂ ${fmt(r.beta)} ` +
      `± ${fmt(r.beta_block_sd)} · Δ ${sign(r.balance_delta)} · band ±${fmt(r.band, 4)} · ${r.verdict}`;
    strip.appendChild(row);
  });

  const negatives = data.records.filter((r) => r.balance_delta < 0).length;
  const note = document.createElement('p');
  note.className = 'plot-note';
  note.style.borderTop = '1px solid var(--paper-edge)';
  note.textContent =
    `${data.n_consistent} consistent, ${data.n_violated} violated. The band is ` +
    `±2×max(block sd, ${data.band_floor}); the ${data.band_floor} floor is the ` +
    `deconvolution's own measured finite-length bias, so departures smaller than ` +
    `the method's error are not called violations. ${negatives} of ` +
    `${data.records.length} deltas are negative — the departures skew one way, ` +
    `which the balance relation alone does not predict.`;
  strip.appendChild(note);
}

/* ============================================================
   3 — Endogeneity
   ============================================================ */

function initEndogeneity(data) {
  const t = theme();
  const s = data.summary;
  const recs = data.records;

  /* --- alpha vs activity, with IQR bars --- */
  const reg = data.activity_regression;
  const xs = recs.map((r) => r.log_n);
  const xMin = Math.min(...xs) - 0.05;
  const xMax = Math.max(...xs) + 0.05;

  const traces1 = [
    {
      type: 'scatter',
      mode: 'markers',
      x: xs,
      y: recs.map((r) => r.alpha_median),
      error_y: {
        type: 'data',
        array: recs.map((r) => r.alpha_iqr / 2),
        color: t.rule,
        thickness: 1,
        width: 3,
      },
      marker: { color: t.plot, size: 8 },
      text: recs.map(
        (r) =>
          `<b>${r.symbol}</b><br>n_events ${int(r.n_events)}<br>` +
          `α̂_median ${fmt(r.alpha_median)}<br>IQR ${fmt(r.alpha_iqr)}<br>` +
          `n̂_CV ${fmt(r.alpha_cv)}<br>converged ${r.n_converged}/6`,
      ),
      hovertemplate: '%{text}<extra></extra>',
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: [xMin, xMax],
      y: [reg.intercept + reg.slope * xMin, reg.intercept + reg.slope * xMax],
      line: { color: t.ink, width: 1.5 },
      hovertemplate:
        `OLS slope ${fmt(reg.slope)} ± ${fmt(reg.stderr)}` +
        `<br>R² ${fmt(reg.r2, 4)} · n ${reg.n}<extra></extra>`,
    },
  ];

  Plotly.react(
    document.getElementById('e-plot'),
    traces1,
    baseLayout(t, {
      xaxis: axis(t, 'log₁₀ aggressor events', { range: [xMin, xMax] }),
      yaxis: axis(t, 'α̂ median (MLE)', { range: [0, 1.02] }),
      shapes: [
        {
          type: 'line',
          xref: 'paper',
          x0: 0,
          x1: 1,
          y0: 1,
          y1: 1,
          line: { color: t.flag, width: 1.5, dash: 'dash' },
          layer: 'below',
        },
      ],
    }),
    PLOT_CONFIG,
  );

  document.getElementById('e-note').textContent =
    `${data.n_successful} symbols, ${data.windows} sub-windows each. Median of the ` +
    `per-symbol medians ${fmt(s.alpha_median_of_medians)}, range ` +
    `${fmt(s.alpha_min)}–${fmt(s.alpha_max)}. OLS on activity: slope ` +
    `${fmt(reg.slope)} (stderr ${fmt(reg.stderr)}), R² ${fmt(reg.r2, 4)}. ` +
    `Dashed line is α = 1, criticality.`;

  /* --- MLE vs count-variance, against y = x --- */
  const traces2 = [
    {
      type: 'scatter',
      mode: 'lines',
      x: [0, 1.02],
      y: [0, 1.02],
      line: { color: t.inkFaint, width: 1, dash: 'dot' },
      hoverinfo: 'skip',
    },
    {
      type: 'scatter',
      mode: 'markers',
      x: recs.map((r) => r.alpha_median),
      y: recs.map((r) => r.alpha_cv),
      marker: { color: t.plot2, size: 8 },
      text: recs.map(
        (r) =>
          `<b>${r.symbol}</b><br>α̂_median (MLE) ${fmt(r.alpha_median)}<br>` +
          `n̂_CV ${fmt(r.alpha_cv)}<br>|diff| ${fmt(r.abs_diff)}`,
      ),
      hovertemplate: '%{text}<extra></extra>',
    },
  ];

  Plotly.react(
    document.getElementById('e-plot2'),
    traces2,
    baseLayout(t, {
      xaxis: axis(t, 'α̂ median — exponential-kernel MLE', { range: [0, 1.02] }),
      yaxis: axis(t, 'n̂ — count-variance', { range: [0, 1.02] }),
    }),
    PLOT_CONFIG,
  );

  document.getElementById('e-note2').textContent =
    `Every point sits above y = x. Median |difference| ` +
    `${fmt(data.agreement.median_abs_diff)}, Pearson correlation ` +
    `${fmt(data.agreement.correlation)} — weak positive, not a strong cross-check.`;

  document.getElementById('e-callout').textContent =
    `The count-variance estimator reads higher than the MLE on ` +
    `${s.n_cv_exceeds_mle} of ${s.n_total} symbols — 100%, not merely on average ` +
    `(median n̂_CV ${fmt(s.cv_median)} against median α̂ ` +
    `${fmt(s.alpha_median_of_medians)}). Two non-exclusive explanations fit a gap ` +
    `in this direction: an exponential kernel fitted to a true power law truncates ` +
    `long-range excitation and understates α, while the count-variance estimator ` +
    `assumes no kernel shape at all; or the fixed 200-second count-variance window ` +
    `is itself biasing that estimator. This data cannot separate them, so the panel ` +
    `establishes high endogeneity and leaves how close to critical unresolved. ` +
    `Averaging the two, or reporting whichever is more publishable, would hide the ` +
    `one thing the comparison actually established.`;
}

/* ============================================================
   4 — Execution
   ============================================================ */

function initExecution(data) {
  const t = theme();
  const summary = data.evaluation.summary;
  const order = ['twap', 'frontloaded', 'reactive'];
  const labels = { twap: 'TWAP', frontloaded: 'Front-loaded', reactive: 'Flow-reactive' };
  const colors = { twap: t.plot, frontloaded: t.band, reactive: t.plot2 };

  const traces = order.map((k) => ({
    type: 'bar',
    name: labels[k],
    x: [labels[k]],
    y: [summary[k].mean_shortfall],
    error_y: {
      type: 'data',
      array: [summary[k].sd_shortfall],
      color: t.inkFaint,
      thickness: 1.2,
      width: 8,
    },
    marker: { color: colors[k] },
    hovertemplate:
      `<b>${labels[k]}</b><br>mean shortfall ${fmt(summary[k].mean_shortfall, 6)}<br>` +
      `sd ${fmt(summary[k].sd_shortfall, 5)}<br>n ${summary[k].n}<extra></extra>`,
  }));

  Plotly.react(
    document.getElementById('x-plot'),
    traces,
    baseLayout(t, {
      xaxis: axis(t, '', { type: 'category' }),
      yaxis: axis(t, 'mean shortfall per unit (± sd)'),
      shapes: [
        {
          type: 'line',
          xref: 'paper',
          x0: 0,
          x1: 1,
          y0: 0,
          y1: 0,
          line: { color: t.inkFaint, width: 1 },
          layer: 'below',
        },
      ],
      bargap: 0.45,
    }),
    PLOT_CONFIG,
  );

  const c = data.calibration;
  document.getElementById('x-note').textContent =
    `Calibrated on ${c.days.join(', ')} over a ` +
    `${c.grid.length}-point grid; chosen lookback ${c.chosen_params.lookback}, ` +
    `pause threshold ${c.chosen_params.pause_threshold}. Evaluated on the disjoint ` +
    `window ${data.evaluation.days.join(', ')} — ${summary.twap.n} cells per ` +
    `schedule across ${data.panel_symbols.length} symbols, both sides, two parent ` +
    `sizes. Error bars are the standard deviation across those cells, not a ` +
    `standard error of the mean.`;

  document.getElementById('x-callout').textContent =
    `Flow-reactive shows the lowest mean shortfall (${fmt(summary.reactive.mean_shortfall, 6)}) ` +
    `and front-loaded the highest (${fmt(summary.frontloaded.mean_shortfall, 6)}), but the ` +
    `reactive-vs-TWAP gap of ${fmt(data.derived.reactive_vs_twap_gap, 5)} is tiny against ` +
    `the dispersion both share (sd ≈ ${fmt(data.derived.mean_sd_twap_reactive, 2)}). This ` +
    `sample does not statistically distinguish them; that apparent edge is consistent with ` +
    `noise. The one clearly resolved effect is front-loaded's variance: sd ` +
    `${fmt(summary.frontloaded.sd_shortfall, 4)} against roughly ` +
    `${fmt(data.derived.mean_sd_twap_reactive, 2)} for the other two, and it holds for every ` +
    `symbol in the panel — trading deterministic own-impact cost for less exposure to noisy ` +
    `drift. Which trade-off is better depends on a risk preference this analysis does not take ` +
    `a position on.`;
}

/* ============================================================
   Scroll reveal + nav state
   ============================================================ */

function initMotion() {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  const items = document.querySelectorAll('.reveal');

  if (reduce.matches || !('IntersectionObserver' in window)) {
    items.forEach((el) => el.classList.add('is-in'));
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          io.unobserve(e.target);
        }
      });
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.08 },
  );
  items.forEach((el) => io.observe(el));
}

function initNav() {
  const links = [...document.querySelectorAll('.site-nav a')];
  const targets = links
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);
  if (!targets.length || !('IntersectionObserver' in window)) return;

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) =>
          a.setAttribute(
            'aria-current',
            String(a.getAttribute('href') === `#${e.target.id}`),
          ),
        );
      });
    },
    { rootMargin: '-45% 0px -50% 0px' },
  );
  targets.forEach((t) => io.observe(t));
}

/* ============================================================
   Boot
   ============================================================ */

function whenPlotlyReady() {
  return new Promise((resolve, reject) => {
    if (window.Plotly) return resolve();
    let waited = 0;
    const tick = setInterval(() => {
      if (window.Plotly) {
        clearInterval(tick);
        resolve();
      } else if ((waited += 60) > 15000) {
        clearInterval(tick);
        reject(new Error('Plotly did not load from the CDN'));
      }
    }, 60);
  });
}

async function boot() {
  initMotion();
  initNav();

  try {
    await whenPlotlyReady();
  } catch (err) {
    ['cs-plot', 'k-plot', 'e-plot', 'e-plot2', 'x-plot'].forEach((id) =>
      failPlot(id, err),
    );
    return;
  }

  // Each section fails independently — one bad slice must not blank the page.
  const jobs = [
    ['crossSection', initCrossSection, ['cs-plot']],
    ['kernels', initKernels, ['k-plot']],
    ['endogeneity', initEndogeneity, ['e-plot', 'e-plot2']],
    ['execution', initExecution, ['x-plot']],
  ];

  await Promise.all(
    jobs.map(async ([key, init, ids]) => {
      try {
        init(await loadJSON(DATA[key]));
      } catch (err) {
        ids.forEach((id) => failPlot(id, err));
      }
    }),
  );

  // Redraw on theme flip so plot ink tracks the page.
  const scheme = window.matchMedia('(prefers-color-scheme: dark)');
  scheme.addEventListener?.('change', () => {
    _resolveCache.clear();
    window.location.reload();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
