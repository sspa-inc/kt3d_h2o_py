# UK_SSPA v2 — Overview

**UK_SSPA v2** (Universal Kriging with Specified Spatial Polynomial and AEM drift) is a Python program for producing spatially interpolated groundwater level maps from point observation data. It implements Universal Kriging with user-specified drift terms, including polynomial trend functions and Analytic Element Method (AEM) linesink potentials derived from river geometry.

---

## 🔬 Validation & Verification

**This software has been formally validated and verified.** The complete V&V report documents 10 validation tests covering all core modules including variogram models, drift terms, anisotropy transformations, and kriging accuracy. **Includes direct comparison against PyKrige** to verify wrapper equivalence and anisotropy consistency.

📥 **[Download V&V Report (PDF)](assets/vv_report.pdf)**
🔍 **[Browse V&V Scripts & Results](validation/)**

---

## What It Does

Given a set of observation wells with measured water levels, UK_SSPA v2:

1. Fits a geostatistical model (variogram) to the spatial structure of the data
2. Optionally applies a coordinate transformation to handle directional anisotropy in spatial correlation
3. Constructs a drift matrix from polynomial terms and/or AEM linesink potentials
4. Solves the Universal Kriging system to produce a predicted water level surface on a regular grid
5. Exports the result as a map, contour shapefile, or point shapefile

The program is driven entirely by a single [`config.json`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/config.json) file — no code changes are required to switch between configurations.

---

## Key Capabilities

### Variogram Models
Four variogram models are supported: **spherical**, **exponential**, **gaussian**, and **linear**. Parameters (sill, range, nugget) are specified in the config. See [`docs/theory/variogram-models.md`](theory/variogram-models.md).

### Polynomial Drift Terms
Regional groundwater gradients can be captured using polynomial drift terms: `linear_x`, `linear_y`, `quadratic_x`, `quadratic_y`. These are rescaled for numerical stability using a factor derived from the variogram sill and data extent. See [`docs/theory/polynomial-drift.md`](theory/polynomial-drift.md).

### AEM Linesink Drift
The hydraulic influence of rivers can be represented as a drift term using the Analytic Element Method. Each river (or group of river segments) generates a potential field that is included as a column in the drift matrix. Scaling factors computed during model training are persisted and reused during grid prediction. See [`docs/theory/aem-linesink.md`](theory/aem-linesink.md).

### Geometric Anisotropy
When spatial correlation is stronger in one direction than another, the coordinate system can be transformed (rotated and scaled) to make the field isotropic before kriging. The angle convention is **azimuth** (clockwise from North). See [`docs/theory/anisotropy.md`](theory/anisotropy.md).

### Leave-One-Out Cross-Validation (LOOCV)
When enabled, the program performs leave-one-out cross-validation and reports RMSE, MAE, and standardized error statistics (Q1, Q2). See [`docs/api/kriging.md`](api/kriging.md).

### Grid Prediction and Export
Predictions are made on a regular grid defined by bounding box and resolution in the config. Output options include a matplotlib map, a contour line shapefile, and an auxiliary point shapefile.

---

## Target Audience

UK_SSPA v2 is designed for:

- **Hydrogeologists** who need to produce groundwater level maps from monitoring well networks, with the ability to incorporate river boundary conditions as drift terms
- **Spatial analysts** working with geostatistical interpolation who need explicit control over drift terms and anisotropy

Users are expected to have:
- A projected point shapefile of observation wells with a numeric water level column
- Knowledge of approximate variogram parameters for their dataset (or results from a variogram fitting tool)
- Optionally: a line shapefile representing rivers, with segment strength values

---

## Architecture

UK_SSPA v2 is a **config-driven, modular pipeline**. All execution logic lives in [`main.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/main.py); each module handles a single concern:

| Module | Responsibility |
|---|---|
| [`data.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/data.py) | Config loading, shapefile I/O, data preparation |
| [`variogram.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/variogram.py) | Variogram model definition and evaluation |
| [`transform.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/transform.py) | Anisotropy coordinate transformation (forward and inverse) |
| [`drift.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/drift.py) | Polynomial drift computation and diagnostics |
| [`AEM_drift.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/AEM_drift.py) | AEM linesink potential computation |
| [`kriging.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/kriging.py) | Universal Kriging model building, prediction, cross-validation |

The pipeline is a linear sequence of stages: load → transform → build drift → build model → predict → export. See [`docs/workflow.md`](workflow.md) for the complete stage-by-stage description.

---

## Documentation Index

| Document | Contents |
|---|---|
| **[Quickstart](quickstart.md)** | Install, configure, and run in 5 minutes |
| **[Configuration Reference](configuration.md)** | Every `config.json` key documented |
| **[Workflow Reference](workflow.md)** | Complete pipeline stage-by-stage |
| **[Data Contracts](data-contracts.md)** | Input shapefile requirements, null handling |
| **[Glossary](glossary.md)** | All domain terms and angle/coordinate conventions |
| **[📥 V&V Report (PDF)](assets/vv_report.pdf)** | **Formal validation report — 10 tests, PyKrige comparison** |
| **[V&V: Tested Behaviors](validation/tested-behaviors.md)** | Summary of all validation test cases |
| **[V&V: Scripts & Results](validation/)** | Browse validation scripts and output |
| **[Theory: Variogram Models](theory/variogram-models.md)** | Model equations and parameter effects |
| **[Theory: Anisotropy](theory/anisotropy.md)** | Coordinate transformation math and conventions |
| **[Theory: Polynomial Drift](theory/polynomial-drift.md)** | Universal Kriging drift formulation |
| **[Theory: AEM Linesink Drift](theory/aem-linesink.md)** | Complex potential formula and scaling |
| **[API: data.py](api/data.md)** | Function signatures and contracts |
| **[API: variogram.py](api/variogram.md)** | Class attributes, methods, validation |
| **[API: transform.py](api/transform.md)** | Transform functions and orientation |
| **[API: drift.py](api/drift.md)** | Drift computation and diagnostics |
| **[API: AEM_drift.py](api/aem_drift.md)** | Linesink potential and scaling |
| **[API: kriging.py](api/kriging.md)** | Model building, prediction, cross-validation |
| **[Example: Ordinary Kriging](examples/ex_ordinary_kriging.md)** | Simplest end-to-end usage |
| **[Example: Linear Drift](examples/ex_linear_drift.md)** | Modeling a regional gradient |
| **[Example: Linesink Drift](examples/ex_linesink_drift.md)** | Incorporating river influence |
| **[Example: Anisotropy](examples/ex_anisotropy.md)** | Directional spatial correlation |
