# Kriging Program Documentation and Benchmarking Plan

## 1. Overview
This document outlines the plan for developing extensive and detailed documentation for the `UK_SSPA v2` kriging program, along with a strategy for benchmarking and validating its core capabilities against established Python packages.

## 2. Documentation Structure

A comprehensive documentation suite should be structured to cater to different user levels, from beginners to advanced users and developers.

### 2.1. Introduction and Overview
*   **Purpose:** What does this program do? (Universal Kriging with specified drift, specifically tailored for spatial data with features like linesinks/rivers).
*   **Key Features:**
    *   Universal Kriging with polynomial drift (linear, quadratic).
    *   Analytic Element Method (AEM) based linesink drift for incorporating river/stream effects.
    *   Anisotropy handling (coordinate transformation).
    *   Leave-One-Out Cross-Validation (LOOCV).
    *   Grid prediction and contour generation.
*   **Target Audience:** Hydrogeologists, spatial data analysts, etc.

### 2.2. Installation and Setup
*   **Dependencies:** List required packages (`numpy`, `pandas`, `geopandas`, `shapely`, `scipy`, `matplotlib`, `pykrige`).
*   **Environment Setup:** Instructions for creating a virtual environment (e.g., `conda` or `venv`) and installing dependencies.

### 2.3. Quickstart Guide
*   A minimal, end-to-end example demonstrating how to run the program with a sample dataset.
*   Basic configuration file (`config.json`) explanation.
*   Running `main.py` and interpreting the output (maps, contours).

### 2.4. User Guide (Detailed Usage)
*   **Configuration (`config.json`):** Deep dive into every parameter.
    *   `data_sources`: Observation wells, line features (rivers).
    *   `variogram`: Model types (spherical, exponential, etc.), sill, range, nugget, anisotropy.
    *   `drift_terms`: Enabling/disabling polynomial and linesink drift.
    *   `grid`: Defining the prediction grid.
    *   `output`: Map generation, contour export, point export.
*   **Data Preparation:** Expected formats for shapefiles (points, lines) and required columns (e.g., water level, elevation).
*   **Understanding Drift:**
    *   Polynomial Drift: How linear and quadratic terms are computed and scaled.
    *   AEM Linesink Drift: Explanation of the analytic element method used for rivers, including the potential function and scaling methods ('adaptive' vs. 'fixed').

### 2.5. Technical Reference and Theory
*   **Universal Kriging Formulation:** Mathematical background of the kriging system solved by the program.
*   **Variogram Models:** Equations for the supported variogram models.
*   **Anisotropy:** Explanation of the coordinate transformation approach (rotation and scaling) used to handle geometric anisotropy.
*   **AEM Linesink Potential:** Detailed derivation and explanation of the complex potential function used in `AEM_drift.py`.
    *   *References:* Cite relevant papers on Analytic Element Method (e.g., Strack, O. D. L. (1989). *Groundwater Mechanics*. Prentice Hall).
*   **Drift Scaling:** Explanation of the `compute_resc` function and why scaling is necessary for numerical stability.

### 2.6. API Reference (Developer Guide)
*   Auto-generated or manually curated documentation for key modules and functions:
    *   `kriging.py`: `build_uk_model`, `predict_on_grid`, `cross_validate`.
    *   `drift.py`: `compute_polynomial_drift`, `verify_drift_physics`.
    *   `AEM_drift.py`: `compute_linesink_drift_matrix`.
    *   `variogram.py`: `variogram` class.
    *   `transform.py`: `get_transform_params`, `apply_transform`.
    *   `data.py`: `load_config`, `prepare_data`.

### 2.7. Examples and Tutorials
*   **Example 1: Ordinary Kriging:** Simple interpolation without drift.
*   **Example 2: Universal Kriging with Linear Drift:** Modeling a regional groundwater gradient.
*   **Example 3: Incorporating River Effects:** Using AEM linesink drift to model a river's influence on the water table.
*   **Example 4: Anisotropy:** Modeling directional dependence in spatial correlation.

## 3. Benchmarking and Validation Plan

To ensure the core capabilities are solid, we will develop a suite of benchmarking tests comparing our implementation against established packages, primarily `PyKrige` (which we wrap, but we need to validate our specific drift formulations) and potentially `scikit-learn` (for basic polynomial regression/drift comparisons).

### 3.1. Test Cases

#### Test Case 1: Ordinary Kriging (Baseline)
*   **Objective:** Verify that our wrapper around `PyKrige` produces identical results for standard Ordinary Kriging.
*   **Setup:** Generate a synthetic dataset with a known variogram.
*   **Comparison:** Compare predictions and variances from our `build_uk_model` (with no drift) against a direct `pykrige.ok.OrdinaryKriging` implementation.
*   **Metrics:** Max absolute difference in predictions and variances (should be near zero).

#### Test Case 2: Universal Kriging with Simple Linear Drift
*   **Objective:** Validate our custom polynomial drift matrix generation and scaling (`drift.py`).
*   **Setup:** Generate a synthetic dataset with a strong linear trend (e.g., $z = ax + by + \text{noise}$).
*   **Comparison:**
    1.  Our program using `drift_terms: {"linear_x": true, "linear_y": true}`.
    2.  `PyKrige`'s built-in `UniversalKriging` with `drift_terms=['regional_linear']`.
*   **Metrics:** Compare predicted surfaces. Note: PyKrige's internal scaling might differ from our `compute_resc`, so exact numerical equivalence might require careful alignment of scaling factors, or we evaluate based on overall surface shape and cross-validation metrics.

#### Test Case 3: Universal Kriging with Quadratic Drift
*   **Objective:** Validate quadratic drift terms.
*   **Setup:** Synthetic dataset with a parabolic trend.
*   **Comparison:** Our program vs. PyKrige with `drift_terms=['point_log']` (or similar, though PyKrige's built-in drifts are limited, we might need to compare against a custom specified drift in PyKrige directly).

#### Test Case 4: Anisotropy Handling
*   **Objective:** Verify our pre-transformation approach (`transform.py`) vs. PyKrige's internal anisotropy handling.
*   **Setup:** Synthetic dataset generated with an anisotropic variogram.
*   **Comparison:**
    1.  Our program: Pre-transform coordinates, run isotropic PyKrige.
    2.  PyKrige: Pass raw coordinates and anisotropy parameters directly to `UniversalKriging`.
*   **Metrics:** Max absolute difference in predictions.

#### Test Case 5: AEM Linesink Drift (Verification)
*   **Objective:** Validate the complex potential calculation in `AEM_drift.py`.
*   **Setup:** A simple scenario with one straight river segment and a few observation points.
*   **Comparison:** Since there isn't a direct equivalent in standard packages, we will validate against analytical solutions or established AEM software (e.g., TimML) if available, or perform rigorous internal consistency checks (e.g., checking potential near the linesink vs. far away).

### 3.2. Implementation of Benchmarks
*   Create a new directory `benchmarks/`.
*   Write Python scripts (e.g., `bench_linear_drift.py`) using `pytest` or a custom benchmarking script to run these comparisons.
*   Use `matplotlib` to generate visual comparisons (difference maps) alongside numerical metrics (RMSE, Max Error).

## 4. Next Steps
1.  Review this plan and confirm the structure and proposed benchmarks.
2.  Begin drafting the documentation sections (starting with Introduction, Installation, and Quickstart).
3.  Implement the benchmarking scripts to validate the core logic.
4.  Refine the Technical Reference section with specific citations for the AEM methodology.