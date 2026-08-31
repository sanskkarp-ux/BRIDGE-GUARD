// BridgeGuard AI -- vanilla JS dashboard. Calls the real POST /predict
// endpoint; nothing here is a hardcoded prediction value. All numbers
// rendered come from the live API response.

const OPTIONS = {
  state: ["Kerala", "Rajasthan", "Uttarakhand", "WestBengal", "Maharashtra", "Gujarat",
          "TamilNadu", "Karnataka", "UttarPradesh", "Bihar", "Assam", "Punjab", "Odisha",
          "Delhi", "MadhyaPradesh"],
  material: ["RCC", "PSC", "Steel", "Masonry", "Timber"],
  structure_type: ["Slab", "Girder", "Box", "Truss", "Arch"],
  road_class: ["NationalHighway", "StateHighway", "DistrictRoad"],
  exposure_condition: ["Mild", "Moderate", "Severe"],
};
// NOTE: mirrors src/api/main.py's VALID_* constants (no schema-introspection
// endpoint exists) -- kept in sync manually, documented in docs/frontend.md.

const CATEGORY_COLOR = {
  Excellent: "#2f8f5b", Good: "#5a9d4a", Fair: "#d9a441", Poor: "#d9642a", Critical: "#c1392b",
};

function populateSelects() {
  for (const [field, values] of Object.entries(OPTIONS)) {
    const el = document.querySelector(`select[name="${field}"]`);
    el.innerHTML = values.map(v => `<option value="${v}">${v}</option>`).join("");
  }
}

function setStatus(kind, message, onRetry) {
  const area = document.getElementById("status-area");
  if (!message) { area.innerHTML = ""; return; }
  if (kind === "loading") {
    area.innerHTML = `<div class="status-msg status-loading"><span class="spinner"></span><span id="loading-text">${message}</span></div>`;
  } else {
    area.innerHTML = `<div class="status-msg status-error">
      <span class="error-icon" aria-hidden="true">&#9888;&#65039;</span>
      <span class="error-text"><strong>ANALYSIS UNAVAILABLE</strong>BridgeGuard could not complete this analysis. ${message}</span>
      <button type="button" class="retry-btn">TRY AGAIN</button>
    </div>`;
    const retryBtn = area.querySelector(".retry-btn");
    if (retryBtn && onRetry) retryBtn.addEventListener("click", onRetry);
  }
}

function buildPayload(form) {
  const fd = new FormData(form);
  const payload = {};
  for (const [key, value] of fd.entries()) {
    if (key === "ever_rehabilitated") continue;
    const isCategorical = /^(state|material|structure_type|road_class|exposure_condition)$/.test(key);
    payload[key] = value === "" ? null : (isCategorical ? value : Number(value));
  }
  payload.ever_rehabilitated = fd.has("ever_rehabilitated");
  return payload;
}

// ---------- animation helpers ----------

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function animateNumber(el, from, to, duration, formatter) {
  const fmt = formatter || (v => Math.round(v));
  if (prefersReducedMotion()) { el.textContent = fmt(to); return; }
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ---------- SHAP text parsing (uses only real API text, just reformatted) ----------

function parseShapExplanation(text) {
  const result = { up: [], down: [] };
  if (!text) return result;
  const lines = text.split("\n");
  let section = null;
  for (const line of lines) {
    if (line.includes("pushing condition UP")) { section = "up"; continue; }
    if (line.includes("pushing condition DOWN")) { section = "down"; continue; }
    const m = line.match(/-\s*(.+?)\s*\(SHAP\s*([+-]?[\d.]+)\)/);
    if (m && section) {
      result[section].push({ name: m[1].trim(), value: parseFloat(m[2]) });
    }
  }
  return result;
}

function renderRiskBars(shapData) {
  const container = document.getElementById("risk-bars");
  const all = [...shapData.down.map(d => ({ ...d, kind: "increase" })),
               ...shapData.up.map(d => ({ ...d, kind: "reduce" }))];
  if (all.length === 0) { container.innerHTML = "<p>No significant risk factors identified.</p>"; return; }
  const maxAbs = Math.max(...all.map(d => Math.abs(d.value)), 0.001);
  const sorted = all.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8);

  container.innerHTML = sorted.map((d, i) => `
    <div class="risk-bar-row">
      <div class="risk-bar-label"><span>${d.name}</span><span>${d.kind === "increase" ? "risk &uarr;" : "risk &darr;"}</span></div>
      <div class="risk-bar-track"><div class="risk-bar-fill ${d.kind}" data-width="${(Math.abs(d.value) / maxAbs * 100).toFixed(0)}" style="transition-delay:${i * 55}ms"></div></div>
    </div>`).join("");

  requestAnimationFrame(() => {
    container.querySelectorAll(".risk-bar-fill").forEach(el => {
      el.style.width = el.dataset.width + "%";
    });
  });
}

function renderShapColumns(shapData) {
  const upList = document.getElementById("shap-up-list");
  const downList = document.getElementById("shap-down-list");
  upList.innerHTML = shapData.up.length
    ? shapData.up.map(d => `<li>${d.name}<span class="shap-val">(${d.value.toFixed(3)})</span></li>`).join("")
    : "<li>None significant</li>";
  downList.innerHTML = shapData.down.length
    ? shapData.down.map(d => `<li>${d.name}<span class="shap-val">(${d.value.toFixed(3)})</span></li>`).join("")
    : "<li>None significant</li>";
}

function drawForecastChart(current, y5, y10) {
  const svg = document.getElementById("forecast-chart");
  const points = [["Now", current], ["+5yr", y5], ["+10yr", y10]].filter(p => p[1] !== null && p[1] !== undefined);
  if (points.length === 0) { svg.innerHTML = ""; return; }

  const W = 400, H = 170, padL = 30, padR = 20, padT = 20, padB = 34;
  const xStep = (W - padL - padR) / Math.max(points.length - 1, 1);
  const yScale = v => H - padB - (v / 9) * (H - padT - padB);

  let path = "", dots = "", labels = "";
  points.forEach((p, i) => {
    const x = padL + i * xStep;
    const y = yScale(p[1]);
    path += (i === 0 ? "M" : "L") + x + "," + y + " ";
    dots += `<circle cx="${x}" cy="${y}" r="5" fill="#d9642a" stroke="#fff" stroke-width="2"><title>${p[0]}: ${p[1]} / 9</title></circle>`;
    labels += `<text x="${x}" y="${H - 12}" font-size="11" text-anchor="middle" fill="#6b655c">${p[0]}</text>`;
    labels += `<text x="${x}" y="${y - 12}" font-size="13" text-anchor="middle" fill="#24211d" font-weight="800">${p[1]}</text>`;
  });

  svg.innerHTML = `
    <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#ece5d8"/>
    <path class="chart-line" d="${path}" fill="none" stroke="#d9a441" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    ${dots}${labels}
  `;
}

function renderRUL(estimate, flag) {
  const el = document.getElementById("rul-content");
  const flagClass = flag === "estimable" ? "flag-estimable" : "flag-uncertain";
  const flagText = flag === "estimable" ? "ESTIMABLE" : "NOT RELIABLY ESTIMABLE";

  const rangeMatch = estimate.match(/approximately\s*(\d+)\s*-\s*(\d+)\s*years/i);
  const medianMatch = estimate.match(/median\s*(\d+)\s*years/i);

  let body, timeline = "";
  if (rangeMatch) {
    const lo = parseInt(rangeMatch[1], 10), hi = parseInt(rangeMatch[2], 10);
    const scaleMax = Math.max(hi * 1.3, 10);
    const loPct = (lo / scaleMax) * 100, hiPct = (hi / scaleMax) * 100;
    body = `<div class="rul-range" style="margin-top:2px">Estimated range</div>
            <div class="rul-number">${lo} &ndash; ${hi} <small style="font-size:1rem;color:var(--text-muted)">years</small></div>`;
    timeline = `<div class="rul-timeline"><div class="rul-timeline-track">
        <div class="rul-timeline-range" data-width="${(hiPct - loPct).toFixed(1)}" style="left:${loPct}%"></div>
        <div class="rul-timeline-dot today"></div>
        <div class="rul-timeline-dot est" style="left:${hiPct}%"></div>
      </div><div class="rul-timeline-labels"><span>Today</span><span>Estimated range</span></div></div>`;
  } else if (medianMatch) {
    const med = parseInt(medianMatch[1], 10);
    const scaleMax = Math.max(med * 1.4, 10);
    const medPct = (med / scaleMax) * 100;
    body = `<div class="rul-number">${med} <small style="font-size:1rem;color:var(--text-muted)">years</small></div>
            <div class="rul-range">Estimated (median)</div>`;
    timeline = `<div class="rul-timeline"><div class="rul-timeline-track">
        <div class="rul-timeline-dot today"></div>
        <div class="rul-timeline-dot est" style="left:${medPct}%"></div>
      </div><div class="rul-timeline-labels"><span>Today</span><span>Median estimate</span></div></div>`;
  } else {
    body = `<div class="rul-uncertain">Not reliably estimable within the observed horizon.</div>`;
  }

  el.innerHTML = `${body}${timeline}<div style="margin-top:8px">Reliability: <span class="rul-flag ${flagClass}">${flagText}</span></div>`;

  requestAnimationFrame(() => {
    el.querySelectorAll(".rul-timeline-range").forEach(r => { r.style.width = r.dataset.width + "%"; });
    el.querySelectorAll(".rul-timeline-dot").forEach(d => { d.style.transform = "translate(-50%, -50%) scale(1)"; });
  });

  const headline = rangeMatch ? `${rangeMatch[1]}–${rangeMatch[2]}` : (medianMatch ? medianMatch[1] : "N/A");
  const headlineUnit = headline === "N/A" ? "" : " yrs";
  document.getElementById("metric-rul").innerHTML = `${headline}<small>${headlineUnit}</small>`;
  document.getElementById("timeline-rul").textContent = headline === "N/A" ? "N/A" : headline + " yrs";
}

function renderResults(data) {
  const resultsEl = document.getElementById("results");
  resultsEl.classList.remove("hidden");

  const scoreCard = document.querySelector(".score-card");
  const forecastCard = document.getElementById("forecast");
  const rulCard = document.getElementById("rul-card");
  const survivalCard = document.getElementById("survival-card");
  const riskCard = document.getElementById("risk");
  const insightsCard = document.getElementById("insights");
  const modelCard = document.querySelector(".model-insights");
  [scoreCard, forecastCard, rulCard, survivalCard, riskCard, insightsCard, modelCard]
    .forEach(el => el && el.classList.remove("is-visible"));
  document.getElementById("score-ring-wrap").classList.remove("glow");

  resultsEl.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });

  const reduced = prefersReducedMotion();
  const stagger = (fn, delay) => { if (reduced) fn(); else setTimeout(fn, delay); };

  stagger(() => {
    scoreCard.classList.add("is-visible");
    const color = CATEGORY_COLOR[data.category] || "#d9a441";
    scoreCard.style.setProperty("--score-color", color);
    animateNumber(document.getElementById("score-value"), 0, data.health_score, 900);
    requestAnimationFrame(() => {
      document.getElementById("score-circle").style.setProperty("--pct", 0);
      setTimeout(() => {
        document.getElementById("score-circle").style.setProperty("--pct", data.health_score);
        document.getElementById("score-ring-wrap").classList.add("glow");
      }, 30);
    });

    const catBadge = document.getElementById("category-badge");
    catBadge.textContent = data.category;
    catBadge.style.background = color + "22";
    catBadge.style.color = color;

    const confBadge = document.getElementById("confidence-badge");
    confBadge.textContent = data.confidence.toUpperCase() + " CONFIDENCE";
    confBadge.className = "confidence-badge conf-" + data.confidence;

    document.getElementById("metric-current").innerHTML = `${data.current_condition ?? "--"}<small>/9</small>`;
    document.getElementById("metric-5yr").innerHTML = `${data["5_year_prediction"] ?? "n/a"}<small>/9</small>`;
    document.getElementById("metric-10yr").innerHTML = `${data["10_year_prediction"] ?? "n/a"}<small>/9</small>`;
    const survivalPct = data.survival_10yr_probability * 100;
    animateNumber(document.getElementById("metric-survival"), 0, survivalPct, 900, v => v.toFixed(1) + "%");
  }, 0);

  stagger(() => {
    forecastCard.classList.add("is-visible");
    document.getElementById("current-condition").textContent = data.current_condition ?? "--";
    document.getElementById("pred-5yr").textContent = data["5_year_prediction"] ?? "n/a";
    document.getElementById("pred-10yr").textContent = data["10_year_prediction"] ?? "n/a";
    drawForecastChart(data.current_condition, data["5_year_prediction"], data["10_year_prediction"]);
  }, 150);

  stagger(() => {
    rulCard.classList.add("is-visible");
    renderRUL(data.rul_estimate, data.rul_reliability_flag);
  }, 270);

  stagger(() => {
    survivalCard.classList.add("is-visible");
    const survivalPct = data.survival_10yr_probability * 100;
    document.getElementById("survival-ring").style.setProperty("--spct", 0);
    setTimeout(() => document.getElementById("survival-ring").style.setProperty("--spct", survivalPct), 30);
    animateNumber(document.getElementById("survival-value"), 0, survivalPct, 900, v => v.toFixed(1) + "%");
  }, 380);

  const shapData = parseShapExplanation(data.shap_explanation);

  stagger(() => {
    riskCard.classList.add("is-visible");
    renderRiskBars(shapData);
  }, 490);

  stagger(() => {
    insightsCard.classList.add("is-visible");
    renderShapColumns(shapData);
  }, 600);

  stagger(() => {
    modelCard.classList.add("is-visible");
    const table = document.getElementById("model-summary-table");
    const summaryRows = [
      ["current condition", data.current_condition],
      ["5-year prediction", data["5_year_prediction"] ?? "n/a"],
      ["10-year prediction", data["10_year_prediction"] ?? "n/a"],
      ["survival probability (10yr)", (data.survival_10yr_probability * 100).toFixed(1) + "%"],
      ["RUL reliability", data.rul_reliability_flag],
      ...Object.entries(data.component_scores || {}).map(([k, v]) => [k.replace(/_/g, " "), v]),
    ];
    table.innerHTML = summaryRows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  }, 700);
}

const LOADING_PHASES = [
  "Evaluating structural condition...",
  "Forecasting deterioration...",
  "Estimating remaining useful life...",
];

async function handleSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById("analyze-btn");
  const btnLabel = btn.querySelector(".btn-label");
  btn.disabled = true;
  btn.classList.add("is-loading");
  if (btnLabel) btnLabel.textContent = "ANALYZING BRIDGE";
  document.getElementById("results").classList.add("hidden");
  setStatus("loading", LOADING_PHASES[0]);

  let phaseIdx = 0;
  const phaseTimer = setInterval(() => {
    phaseIdx = (phaseIdx + 1) % LOADING_PHASES.length;
    const el = document.getElementById("loading-text");
    if (el) el.textContent = LOADING_PHASES[phaseIdx];
  }, 900);

  const retry = () => handleSubmit(e);
  const payload = buildPayload(form);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.status === 422) {
      const err = await res.json();
      setStatus("error", "Invalid input: " + JSON.stringify(err.detail), retry);
      return;
    }
    if (!res.ok) {
      setStatus("error", `Prediction failed (HTTP ${res.status}).`, retry);
      return;
    }

    const data = await res.json();
    setStatus(null);
    renderResults(data);
  } catch (err) {
    setStatus("error", "Could not reach the BridgeGuard API. Is the server running? (" + err.message + ")", retry);
  } finally {
    clearInterval(phaseTimer);
    btn.disabled = false;
    btn.classList.remove("is-loading");
    if (btnLabel) btnLabel.textContent = "ANALYZE BRIDGE";
  }
}

// ---------- scroll / motion chrome ----------

function initScrollProgress() {
  const bar = document.getElementById("scroll-progress");
  if (!bar) return;
  function update() {
    const h = document.documentElement;
    const height = h.scrollHeight - h.clientHeight;
    bar.style.width = (height > 0 ? (h.scrollTop / height) * 100 : 0) + "%";
  }
  window.addEventListener("scroll", update, { passive: true });
  update();
}

function initNavbarScrollState() {
  const nav = document.querySelector(".navbar");
  if (!nav) return;
  function update() { nav.classList.toggle("scrolled", window.scrollY > 12); }
  window.addEventListener("scroll", update, { passive: true });
  update();
}

function initActiveNav() {
  const navLinks = document.querySelectorAll(".navbar-nav a");
  if (!navLinks.length || !("IntersectionObserver" in window)) return;
  const map = {};
  navLinks.forEach(a => { map[a.getAttribute("href").slice(1)] = a; });
  const targets = Object.keys(map).map(id => document.getElementById(id)).filter(Boolean);
  if (!targets.length) return;
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(a => a.classList.remove("active"));
        const link = map[entry.target.id];
        if (link) link.classList.add("active");
      }
    });
  }, { rootMargin: "-40% 0px -50% 0px", threshold: 0 });
  targets.forEach(t => observer.observe(t));
}

function initBackToTop() {
  const btn = document.getElementById("back-to-top");
  if (!btn) return;
  window.addEventListener("scroll", () => {
    btn.classList.toggle("visible", window.scrollY > 500);
  }, { passive: true });
  btn.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  });
}

function initHeroParallax() {
  const bg = document.querySelector(".hero-bg");
  if (!bg || prefersReducedMotion() || window.innerWidth < 760) return;
  let ticking = false;
  function update() {
    bg.style.transform = `translateY(${Math.min(window.scrollY * 0.15, 60)}px)`;
    ticking = false;
  }
  window.addEventListener("scroll", () => {
    if (!ticking) { requestAnimationFrame(update); ticking = true; }
  }, { passive: true });
}

function initScrollReveal() {
  // Cards inside #results are revealed by the staggered sequence in
  // renderResults() instead -- excluded here so this observer (which
  // starts watching before any prediction exists) can't fire early and
  // short-circuit that sequence once #results becomes visible.
  const els = Array.from(document.querySelectorAll(".io-reveal, .io-reveal-scale, .io-reveal-left, .io-reveal-right"))
    .filter(el => !el.closest("#results"));
  if (!els.length) return;
  if (prefersReducedMotion() || !("IntersectionObserver" in window)) {
    els.forEach(el => el.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });
  els.forEach(el => observer.observe(el));
}

populateSelects();
document.getElementById("predict-form").addEventListener("submit", handleSubmit);
initScrollProgress();
initNavbarScrollState();
initActiveNav();
initBackToTop();
initHeroParallax();
initScrollReveal();
