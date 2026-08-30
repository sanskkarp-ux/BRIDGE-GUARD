# Research Review — Bridge Deterioration & Condition Prediction

This document summarizes, in our own words, how the field has approached bridge deterioration modeling, what data and methods have been used, where those approaches fall short, and what BridgeGuard AI intends to do differently or better. It is not a literature copy — it is a working understanding we'll revise as we learn more during implementation.

---

## 1. The classical approach: Markov chain transition models

**What it is.** The dominant approach in actual state-DOT practice is not machine learning at all — it's **Markov chain deterioration modeling**, embedded in AASHTOWare Bridge Management (BrM), the successor to the older Pontis system. The idea: represent a bridge component's condition as a discrete state (0–9), and estimate a **transition probability matrix** describing the chance of moving from one condition state to a worse one within a given time step (typically one year). Life-cycle cost analysis and maintenance scheduling are then built on top of these transition probabilities.

**Where it falls short.** The most commonly cited weakness, echoed across multiple recent papers we found (e.g., comparative studies of Markovian bridge deterioration approaches), is that the *classic* Pontis-style model uses **constant transition probabilities** — the same probability of moving from state 6 to state 5 whether the bridge is 10 years old or 80 years old, and regardless of traffic load or climate. That's a real simplification: deterioration is not a memoryless, time-invariant process in reality. Later refinements (state- and age-dependent Markov models, clustering by superstructure type before fitting separate transition matrices) partially address this by conditioning the matrix on bridge characteristics or age bands, but they're still fundamentally discretizing a continuous covariate-driven process into a fixed matrix.

**What we take from this.** The Markov framework is exactly the right conceptual scaffold for the RUL layer (§6/§7 of the architecture) — but instead of a hand-fit transition matrix, we intend to let the transition probabilities be a *function of covariates* (age, traffic, material, climate) via a discrete-time hazard/survival model, which is a more flexible generalization of the same idea rather than a wholesale departure from established practice.

---

## 2. Machine learning approaches on NBI data

Several recent studies (2023–2025) apply supervised ML directly to NBI-derived features to predict condition ratings, most commonly **deck condition**, since it's the most frequently studied of the three components:

- Comparative work evaluating **Random Forest, XGBoost, and Artificial Neural Networks** on U.S. bridge data for deck condition prediction found ensemble tree methods (Random Forest and XGBoost) competitive with or better than ANN — notable because it argues against defaulting to deep learning, consistent with our own project instruction to establish classical baselines first.
- A case study focused on **Ohio bridges** compared multiple ML algorithms (decision trees, ANN, k-NN, logistic regression, SVM, Random Forest, XGBoost) specifically for bridge deck deterioration, finding ensemble methods (Random Forest, XGBoost, and gradient boosting variants) consistently outperform single-model approaches, and that **feature selection matters** — models did meaningfully better with a curated feature set than with the raw column dump.
- A broader "bridging the gap" comparative study explicitly evaluates **explainable AI approaches** alongside traditional and hybrid ML for bridge condition prediction, which validates our project's SHAP-based explainability requirement as an active area of interest, not an afterthought bolted onto a black-box model.
- One recent study explicitly integrates **environmental conditions** (climate variables) into ML models for bridge deterioration prediction, using entropy-based mutual information analysis to evaluate how much traffic, location, and age variables actually contribute — a useful methodological pattern we can borrow for our own feature-importance validation instead of assuming every proposed feature (e.g., freeze-thaw cycles) matters just because it's physically plausible.
- Work using ML with **missing data scenarios** specifically for bridge deterioration is directly relevant, since NBI's real-world fill rates are uneven across states/years (see `data_sources.md`) — this tells us missingness-robust modeling isn't a hypothetical concern, it's a documented issue others have had to solve too.

**What we take from this.** The published performance ordering (ensemble trees ≥ ANN, feature curation matters, ANN doesn't obviously win) directly supports our project's instruction to start with Random Forest / Gradient Boosting / XGBoost and treat deep learning as unjustified by default rather than a natural next step.

---

## 3. Survival analysis and remaining useful life (RUL)

We found comparatively little published work applying formal **survival analysis** directly to bridge RUL using NBI-scale panel data — most of the RUL/survival literature we found in general ML is from other domains (e.g., mechanical component RUL prediction, such as censoring-aware survival methods for bearing degradation). That domain-general survival methodology (handling right-censored observations — components that haven't failed yet at the time of the snapshot) transfers conceptually to bridges: **most bridges in any NBI snapshot have not yet reached "poor" condition**, so they are right-censored observations for a "time to poor condition" survival model, exactly as an unfailed bearing is censored in mechanical RUL work.

**What we take from this.** RUL for bridges is a genuine research gap where our project can do something more rigorous than a naive "years until threshold" heuristic — by treating it explicitly as a censored survival problem rather than pretending every bridge in the dataset has a known "time to failure."

---

## 4. Structural health monitoring (SHM) — noted but out of scope

Sensor-based SHM (accelerometers, strain gauges, fiber optics) is a large, active research area for individual instrumented bridges, but it's a fundamentally different data regime from what we have: dense continuous time series on a small number of specially-instrumented structures, versus sparse biennial administrative inspections on hundreds of thousands of ordinary bridges. We are not building an SHM system. We note this distinction explicitly so nobody (including the eventual app's users) mistakes BridgeGuard AI's condition-trend output for the kind of real-time structural signal SHM systems provide.

---

## 5. Climate and traffic loading effects

The environmental-integration study noted above is the most directly relevant precedent for our climate feature plan — it confirms that climate-conditioned ML deterioration models are a live research direction, not a speculative addition, and that formal feature-importance validation (rather than assuming climate matters) is the right way to justify keeping or dropping climate features once we have real fitted models.

---

## 6. What BridgeGuard AI aims to improve on

Based on the above, our differentiation is not "use a fancier model" — it's:

1. **Covariate-conditioned transition/hazard modeling** instead of a fixed Markov transition matrix, addressing the most commonly cited weakness of the industry-standard approach.
2. **Explicit survival treatment of RUL** with right-censoring handled correctly, rather than a deterministic "years to threshold" number — directly serving the uncertainty requirement in the project brief.
3. **Rigorous temporal validation and leakage discipline** (see `docs/model_validation.md`, `docs/data_leakage.md`) — several of the papers reviewed do not describe their train/test splitting strategy in enough detail to confirm they avoided bridge-level or temporal leakage; we intend to document ours explicitly enough that it's auditable.
4. **Explainability as a first-class output**, not a post-hoc add-on, consistent with where the field's more recent (2025) comparative work is already heading.
5. **Honest handling of the SNBI 2025 data transition**, which is a very recent, real, and currently under-documented practical issue that a lot of NBI-based research predates.

We are not claiming to out-perform the field on raw accuracy in Phase 1 — the goal of the baseline phase is to establish a correct, leakage-free, well-validated foundation, matching what the literature already shows works (ensemble trees), before attempting any of the above improvements.
