# India Data Strategy

BridgeGuard AI's target shifted from U.S. NBI data to Indian public bridges (see conversation history — no separate research doc exists for this decision itself; this file documents the India data strategy only). This is the current, active data plan. The prior India-wide dataset survey (IBMS, NHAI BICRS, multiple state PWDs, data.gov.in, IMD, CWC/WRIS, research papers) informed this decision but is not repeated here — this doc only covers the chosen path forward.

---

## Primary Dataset: Uttarakhand PWD InfraMgt

**URL**: `https://pwduk.in/im/` — individual bridge records at `https://pwduk.in/im/bridgReport/<ID>`
**Parent organization**: Public Works Department, Government of Uttarakhand
**Official department site**: `https://pwd.uk.gov.in/`

### Why it's useful
Of every Indian source checked, this is the only one confirmed to expose **individual, real, richly-detailed bridge records** — comparable in depth to NBI. A live record we viewed (Bridge ID 2444, Chandrabhaga bridge) contained: name, structure number, river feature, nearest town, latitude/longitude, construction year, inventory year, full geometry (length, spans, span arrangement, lanes, widths, skew angle), flood levels (HFL/OFL/LWL), design discharge, design scour level at pier/abutments, exposure condition, seismic factor, design loading class, superstructure type/material/wearing coat/bearings, substructure foundation/pier/abutment type, bank/floor protection work, and a corrosion-protection-treatment field. No other Indian source surveyed came close to this field depth.

### What we know
- Individual bridge record pages are **viewable without login**.
- Records are addressed by a numeric ID (`bridgReport/<ID>`); IDs at least into the low thousands exist.
- Each record shows a **rating field** (e.g., "Rating: 15.54%"), a **"Data finalized On"** date, and a separate **"Last Rated On"** date — meaning the record has been touched/re-rated at least once after initial data entry.
- No login was required to reach this data through normal navigation from a shared search result link.

### What remains unknown
- **What the rating percentage actually means** (methodology, scale direction, what "good" vs. "bad" looks like) — not explained on the page itself.
- **Whether a true historical time series exists** — the "Last Rated On" date differing from "Data finalized On" hints at re-rating, but we did not find (and did not look for, per instructions not to scrape) any page exposing multiple past ratings for the same bridge ID.
- **Total number of bridges covered** — no bridge list/dashboard URL was found; only the individual-record URL pattern is confirmed.
- **Whether bulk export, an API, or a research-data request process exists** — not found via light probing of a few guessed URL paths (all 404).
- **Terms of use / licensing** — no license or terms-of-use statement was found on the pages viewed. Public viewability does not imply permission for bulk reuse.

### Access / permission requirements
No bulk data was downloaded and no scraping was attempted, per instructions. To legally use this data beyond casual individual-record viewing, we need **explicit written permission from PWD Uttarakhand** — most plausibly through:
- The RTI (Right to Information) route: `rti.pwd@uk.gov.in`
- Or a direct research-data request to the Chief Engineer (HQ): R.S. Sayana, `cehq.pwduk@gov.in` / `cehqpwduk@gmail.com`, 0135-2531424
- General contact page: `https://pwd.uk.gov.in/contact-us/`

---

## Backup Strategy: IMD + CWC/India-WRIS Environmental Data

- **IMD** (India Meteorological Department): gridded rainfall (1901–2024, 0.25°) and temperature (1951–present, 1°) — confirmed public, downloadable, no permission needed.
- **CWC / India-WRIS**: river discharge and water-level station data — confirmed public for "unclassified" stations, free download, no permission needed.

These don't depend on PWD Uttarakhand access and can be built regardless of how the primary-dataset request goes. They provide the climate/flood-exposure feature layer, joinable to any bridge's lat-long (which the InfraMgt records do include).

---

## Long-Term: MoRTH IBMS/UBMS or NHAI BICRS

If a relationship with PWD Uttarakhand proves workable, the same request template (research/RTI ask, framed as an academic/portfolio ML project) can later be extended to MoRTH (IBMS/UBMS, ~1.7 lakh structures nationally) or NHAI (BICRS, bi-annual inspections mandated from April 2025 — but too new to have multi-year history yet). These are larger-scale, higher-effort asks and are not the current priority.

---

## How This Fits Into BridgeGuard

- **If permission is granted** and the data turns out to include genuine repeat observations: this becomes the core panel dataset, replacing NBI's role in the original architecture (same modeling plan — classical baselines → deterioration forecasting → survival-based RUL — just on Uttarakhand bridges instead of U.S. states).
- **If only a single snapshot per bridge is available**: the project scope narrows to cross-sectional condition prediction (current risk given structure/age/environment), with RUL only possible via the cruder cross-sectional age-vs-condition fallback noted in the earlier India research pass — not true survival analysis.
- **Either way**, the IMD/CWC environmental layer is buildable now and used in both scenarios.

---

## WHAT I NEED TO DO

- [ ] Email PWD Uttarakhand to request research/bulk access to InfraMgt bridge data — use either:
  - RTI route: `rti.pwd@uk.gov.in`, or
  - Direct request to Chief Engineer (HQ) R.S. Sayana: `cehq.pwduk@gov.in` / `cehqpwduk@gmail.com` (phone: 0135-2531424)
- [ ] In the request, ask specifically for: (a) bulk/CSV or API access to bridge inventory + rating records, (b) an explanation of the rating field's methodology and scale, (c) whether historical/repeated rating records exist per bridge, and (d) permission terms for use in a student/research ML project.
- [ ] Wait for a response before any bulk download or scraping of `pwduk.in`.
