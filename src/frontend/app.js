// BridgeGuard AI -- vanilla JS dashboard. Calls the real POST /predict
// endpoint; nothing here is a hardcoded prediction value.

const OPTIONS = {
  state: ["Kerala", "Rajasthan", "Uttarakhand", "WestBengal", "Maharashtra", "Gujarat",
          "TamilNadu", "Karnataka", "UttarPradesh", "Bihar", "Assam", "Punjab", "Odisha",
          "Delhi", "MadhyaPradesh"],
  material: ["RCC", "PSC", "Steel", "Masonry", "Timber"],
  structure_type: ["Slab", "Girder", "Box", "Truss", "Arch"],
  road_class: ["NationalHighway", "StateHighway", "DistrictRoad"],
  exposure_condition: ["Mild", "Moderate", "Severe"],
};
// NOTE: these lists mirror src/api/main.py's VALID_* constants. Kept in sync
// manually (no schema-introspection endpoint exists) -- documented in docs/frontend.md.

const CATEGORY_COLOR = {
  Excellent: "#1b998b", Good: "#2f8f5b", Fair: "#c9a13b", Poor: "#c9622b", Critical: "#b3261e",
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
  const cls = kind === "error" ? "status-error" : "status-loading";
  area.innerHTML = `<div class="status-msg ${cls}">${message}</div>`;
}

function buildPayload(form) {
  const fd = new FormData(form);
  const payload = {};
  for (const [key, value] of fd.entries()) {
    if (key === "ever_rehabilitated") continue;
    payload[key] = value === "" ? null : (isNaN(value) || key.match(/^(state|material|structure_type|road_class|exposure_condition)$/) ? value : Number(value));
  }
  payload.ever_rehabilitated = fd.has("ever_rehabilitated");
  return payload;
}

function renderResults(data) {
  document.getElementById("results").classList.remove("hidden");

  document.getElementById("score-value").textContent = data.health_score;
  document.getElementById("score-circle").style.setProperty("--pct", data.health_score);
  const color = CATEGORY_COLOR[data.category] || "#1b998b";
  document.getElementById("score-circle").style.background =
    `conic-gradient(${color} calc(${data.health_score} * 1%), #e6eaee 0)`;
  const catBadge = document.getElementById("category-badge");
  catBadge.textContent = data.category;
  catBadge.style.background = color + "22";
  catBadge.style.color = color;
  document.getElementById("confidence-value").textContent = data.confidence;

  document.getElementById("current-condition").textContent = data.current_condition ?? "--";
  document.getElementById("pred-5yr").textContent = data["5_year_prediction"] ?? "n/a";
  document.getElementById("pred-10yr").textContent = data["10_year_prediction"] ?? "n/a";
  drawForecastChart(data.current_condition, data["5_year_prediction"], data["10_year_prediction"]);

  const rulEl = document.getElementById("rul-content");
  const flagClass = data.rul_reliability_flag === "estimable" ? "flag-estimable" : "flag-uncertain";
  const flagText = data.rul_reliability_flag === "estimable" ? "Estimable" : "Not reliably estimable";
  rulEl.innerHTML = `
    <div class="rul-line"><strong>Estimate:</strong> ${data.rul_estimate}</div>
    <div class="rul-line"><span class="rul-flag ${flagClass}">${flagText}</span></div>
    <div class="rul-line"><strong>10-year survival probability:</strong> ${(data.survival_10yr_probability * 100).toFixed(1)}%</div>
  `;

  const riskList = document.getElementById("risk-factors-list");
  riskList.innerHTML = (data.top_risk_factors || []).map(f => `<li>${f}</li>`).join("")
    || "<li>No significant risk factors identified.</li>";

  document.getElementById("shap-explanation").textContent = data.shap_explanation || "Not available.";

  const table = document.getElementById("model-summary-table");
  const rows = Object.entries(data.component_scores || {})
    .map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td>${v}</td></tr>`).join("");
  table.innerHTML = rows;
}

function drawForecastChart(current, y5, y10) {
  const svg = document.getElementById("forecast-chart");
  const points = [["Now", current], ["+5yr", y5], ["+10yr", y10]].filter(p => p[1] !== null && p[1] !== undefined);
  if (points.length === 0) { svg.innerHTML = ""; return; }

  const W = 400, H = 160, padL = 30, padR = 20, padT = 15, padB = 30;
  const xStep = (W - padL - padR) / Math.max(points.length - 1, 1);
  const yScale = v => H - padB - (v / 9) * (H - padT - padB);

  let path = "";
  let dots = "";
  let labels = "";
  points.forEach((p, i) => {
    const x = padL + i * xStep;
    const y = yScale(p[1]);
    path += (i === 0 ? "M" : "L") + x + "," + y + " ";
    dots += `<circle cx="${x}" cy="${y}" r="4" fill="#1b4965"></circle>`;
    labels += `<text x="${x}" y="${H - 10}" font-size="11" text-anchor="middle" fill="#5a6472">${p[0]}</text>`;
    labels += `<text x="${x}" y="${y - 10}" font-size="12" text-anchor="middle" fill="#1b4965" font-weight="700">${p[1]}</text>`;
  });

  svg.innerHTML = `
    <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#e0e4e9"/>
    <path d="${path}" fill="none" stroke="#1b998b" stroke-width="2.5"/>
    ${dots}${labels}
  `;
}

async function handleSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  document.getElementById("results").classList.add("hidden");
  setStatus("loading", "Analyzing bridge -- calling prediction pipeline...");

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
  } catch (err) {
    setStatus("error", "Could not reach the BridgeGuard API. Is the server running? (" + err.message + ")");
  } finally {
    btn.disabled = false;
  }
}

populateSelects();
document.getElementById("predict-form").addEventListener("submit", handleSubmit);
