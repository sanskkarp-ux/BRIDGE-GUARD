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

function setStatus(kind, message) {
  const area = document.getElementById("status-area");
  if (!message) { area.innerHTML = ""; return; }
  if (kind === "loading") {
    area.innerHTML = `<div class="status-msg status-loading"><span class="spinner"></span>${message}</div>`;
  } else {
    area.innerHTML = `<div class="status-msg status-error">${message}</div>`;
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

function animateNumber(el, from, to, duration, formatter) {
  const start = performance.now();
  const fmt = formatter || (v => Math.round(v));
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

  container.innerHTML = sorted.map(d => `
    <div class="risk-bar-row">
      <div class="risk-bar-label"><span>${d.name}</span><span>${d.kind === "increase" ? "risk &uarr;" : "risk &darr;"}</span></div>
      <div class="risk-bar-track"><div class="risk-bar-fill ${d.kind}" data-width="${(Math.abs(d.value) / maxAbs * 100).toFixed(0)}"></div></div>
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
    dots += `<circle cx="${x}" cy="${y}" r="5" fill="#d9642a" stroke="#fff" stroke-width="2"></circle>`;
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

  let body;
  if (rangeMatch) {
    body = `<div class="rul-range" style="margin-top:2px">Estimated range</div>
            <div class="rul-number">${rangeMatch[1]} &ndash; ${rangeMatch[2]} <small style="font-size:1rem;color:var(--text-muted)">years</small></div>`;
  } else if (medianMatch) {
    body = `<div class="rul-number">${medianMatch[1]} <small style="font-size:1rem;color:var(--text-muted)">years</small></div>
            <div class="rul-range">Estimated (median)</div>`;
  } else {
    body = `<div class="rul-uncertain">Not reliably estimable within the observed horizon.</div>`;
  }

  el.innerHTML = `${body}<div style="margin-top:8px">Reliability: <span class="rul-flag ${flagClass}">${flagText}</span></div>`;
}

function renderResults(data) {
  document.getElementById("results").classList.remove("hidden");

  const color = CATEGORY_COLOR[data.category] || "#d9a441";
  document.getElementById("score-circle").style.setProperty("--score-color", color);
  animateNumber(document.getElementById("score-value"), 0, data.health_score, 900);
  requestAnimationFrame(() => {
    document.getElementById("score-circle").style.setProperty("--pct", 0);
    setTimeout(() => document.getElementById("score-circle").style.setProperty("--pct", data.health_score), 30);
  });

  const catBadge = document.getElementById("category-badge");
  catBadge.textContent = data.category;
  catBadge.style.background = color + "22";
  catBadge.style.color = color;

  const confBadge = document.getElementById("confidence-badge");
  confBadge.textContent = data.confidence.toUpperCase() + " CONFIDENCE";
  confBadge.className = "confidence-badge conf-" + data.confidence;

  document.getElementById("current-condition").textContent = data.current_condition ?? "--";
  document.getElementById("pred-5yr").textContent = data["5_year_prediction"] ?? "n/a";
  document.getElementById("pred-10yr").textContent = data["10_year_prediction"] ?? "n/a";
  drawForecastChart(data.current_condition, data["5_year_prediction"], data["10_year_prediction"]);

  renderRUL(data.rul_estimate, data.rul_reliability_flag);

  const survivalPct = data.survival_10yr_probability * 100;
  document.getElementById("survival-ring").style.setProperty("--spct", 0);
  setTimeout(() => document.getElementById("survival-ring").style.setProperty("--spct", survivalPct), 30);
  animateNumber(document.getElementById("survival-value"), 0, survivalPct, 900, v => v.toFixed(1) + "%");

  const shapData = parseShapExplanation(data.shap_explanation);
  renderRiskBars(shapData);
  renderShapColumns(shapData);

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
}

async function handleSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  document.getElementById("results").classList.add("hidden");
  setStatus("loading", "Analyzing bridge -- running prediction pipeline...");

  const payload = buildPayload(form);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.status === 422) {
      const err = await res.json();
      setStatus("error", "Invalid input: " + JSON.stringify(err.detail));
      return;
    }
    if (!res.ok) {
      setStatus("error", `Prediction failed (HTTP ${res.status}).`);
      return;
    }

    const data = await res.json();
    setStatus(null);
    renderResults(data);
    document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setStatus("error", "Could not reach the BridgeGuard API. Is the server running? (" + err.message + ")");
  } finally {
    btn.disabled = false;
  }
}

populateSelects();
document.getElementById("predict-form").addEventListener("submit", handleSubmit);
