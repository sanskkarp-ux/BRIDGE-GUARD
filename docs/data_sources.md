# Data Sources

This document records every dataset BridgeGuard AI is built on, why it was chosen, what's usable in it, and what isn't. It is a living document — update it whenever a dataset decision changes, rather than leaving this out of sync with the pipeline.

---

## 1. Primary Dataset: FHWA National Bridge Inventory (NBI)

**Source**: U.S. Federal Highway Administration
**URL**: https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
**Dataset years available**: 1992–2025 (one snapshot per year, one file per state, plus combined files)
**License / access**: Public domain, freely downloadable, no registration required. Data is submitted annually by states, federal agencies, and tribal governments under the National Bridge Inspection Standards (23 CFR Part 650 Subpart C), and FHWA publishes the finalized snapshot by June 15 of each year.
**Format**: Delimited ASCII/CSV, comma-separated, single-quote text qualifier. Documented in the *Recording and Coding Guide for the Structure Inventory and Appraisal of the Nation's Bridges* (legacy, pre-2025) and the *Specifications for the National Bridge Inventory* (SNBI, FHWA-HIF-22-017, in effect from January 2025).

### Why this dataset
It's the only source that gives us, for nearly every public highway bridge in the country, a multi-decade longitudinal record of the same three variables we need as targets (deck, superstructure, substructure condition), alongside the structural, traffic, and geometric attributes needed as features. No other public dataset has this combination of coverage, history depth, and standardization.

### Critical structural fact: the 2025 SNBI transition
States began submitting data in the new SNBI format starting January 2025 (full compliance mandated by 2028). This is not a cosmetic change — item numbers were renamed and some fields were split into finer granularity:

| Concept | Legacy item (1992–2024) | SNBI item (2025+) |
|---|---|---|
| Deck condition | Item 58 | B.C.01 |
| Superstructure condition | Item 59 | B.C.02 |
| Substructure condition | Item 60 | B.C.03 |
| Culvert condition | Item 62 | B.C.04 |

Additionally, SNBI introduces separately-rated bearings, joints, and railings that were previously folded into the superstructure rating under the legacy scheme. **This means a naive concatenation of 1992–2025 files will silently misalign columns at the boundary.** The ingestion layer must explicitly map legacy → SNBI item names before treating the panel as continuous. Until we've verified the harmonization is correct, the pilot and initial model development should stay within the legacy-coding years (1992–2024) to avoid building on a broken join.

### Relevant columns (legacy item numbers, primary set)

**Identification / join key**
- Structure Number — unique bridge identifier, the join key across inspection years
- State Code, County Code
- Item 27 — Year Built
- Item 16 / Item 17 — Latitude / Longitude

**Targets**
- Item 58 — Deck condition (0–9)
- Item 59 — Superstructure condition (0–9)
- Item 60 — Substructure condition (0–9)
- Item 62 — Culvert condition (0–9; applies instead of deck/super/sub for culvert-type structures)

**Structural / geometric features**
- Item 43A — Structure material (main span)
- Item 43B — Structure type (main span)
- Item 107 — Deck structure type
- Item 108A — Wearing surface type
- Item 108C — Deck protection
- Item 45 — Number of main spans
- Item 46 — Number of approach spans
- Item 48 — Maximum span length
- Item 49 — Structure length
- Item 52 — Deck width, out-to-out
- Item 28A — Lanes on structure
- Item 28B — Lanes under structure

**Traffic / loading**
- Item 29 — Average Daily Traffic (ADT)
- Item 109 — ADTT as a *percentage* of ADT (not a raw truck count — ADTT must be derived as `ADT × (Item 109 / 100)`)

**Maintenance history**
- Item 106 — Year reconstructed (most recent major rehab/reconstruction)

**Risk context**
- Item 113 — Scour critical indicator

### Target variable definition
NBI has **no single "overall condition" field** in the legacy scheme. FHWA's own condition-measure computation (FHWA-HIF-18-023) uses the **minimum of deck, superstructure, and substructure** ratings as the basis for "poor" classification — we adopt that same convention for our "overall condition" derived target rather than inventing a new one.

### Missing-value concerns
- Column fill rates vary by state and by year — some states have historically left optional items sparse. This must be measured empirically per state/year during the pilot, not assumed.
- Item 106 (year reconstructed) is 0 or blank for bridges that have never been reconstructed — this is a legitimate "no rehab" signal, not a missing value, and must not be imputed as if it were.
- Culvert-type structures have Item 62 populated and Items 58/59/60 coded "N" (not applicable) — these records need a separate handling path, not row-wise deletion.

### Potential leakage
See `docs/data_leakage.md` for full treatment. Summary: Item 106 must be truncated at the prediction cutoff year; scour critical updates may co-occur with condition rating updates in the same inspection cycle; state-level methodology changes can look like deterioration signal but are actually reporting artifacts.

### Limitations
- Ratings are inspector judgment (0–9 ordinal scale), not sensor measurements — inter-inspector and inter-state variability is real noise.
- Observations are biennial at best for most structures — sub-cycle events (e.g., a mid-cycle scour event) aren't captured.
- No true "failure" event exists in the data — bridges are replaced/rehabilitated administratively, so any RUL derived from this data means "time to reach a poor-condition threshold," not "time to structural failure."

---

## 2. Supplementary Dataset (candidate, not yet integrated): LTBP InfoBridge

**Source**: FHWA Long-Term Bridge Performance (LTBP) Program
**Purpose**: Deeper element-level condition and inspection detail for a curated subset of bridges (richer than NBI's three headline ratings, but far smaller coverage).
**Status**: Not used in Phase 1. Candidate for a later phase if element-level granularity becomes necessary beyond what NBI's deck/superstructure/substructure ratings provide.

---

## 3. Supplementary Dataset (candidate, not yet integrated): Climate Data

**Source**: PRISM Climate Group (Oregon State University), built from NOAA and partner station networks (~20,000 precipitation stations, ~10,000 temperature stations, ~4,000 dew point/VPD stations).
**Alternative**: NOAA NCEI "Climate at a Glance" county-level mapping, as a simpler fallback if the PRISM gridded join proves too heavy for a first pass.
**Purpose**: Temperature, precipitation, and humidity features by bridge location (joined via Item 16/17 lat-lon or county code).
**Important note**: freeze-thaw cycle counts are **not published directly** by PRISM or NOAA as a ready-made "by county" dataset — this has to be a *derived* feature (counting daily temperature crossings of 0°C from PRISM daily records), not a raw import. This should be flagged explicitly wherever freeze-thaw appears as a feature, so nobody mistakes it for an official government statistic.
**Status**: Not used in Phase 1. To be integrated once the core NBI panel and baseline models are working, per the roadmap.

---

## 4. Datasets deliberately excluded from Phase 1

- **USGS StreamStats / hydraulic scour data** — Item 113 (Scour Critical) inside NBI itself may be sufficient to start; a dedicated hydraulic dataset is a later enhancement if scour proves to matter in feature importance results, not a Phase 1 commitment.
- Any dataset not sourced from an authoritative government or research body — per project instruction, we do not invent or use unverified datasets.
