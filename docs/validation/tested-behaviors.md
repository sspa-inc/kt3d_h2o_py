# Tested Behaviors Catalog

**Task 7.1 — Phase 7: Test-to-Documentation Mapping**

This document maps every documented behavioral claim to the test file and function that verifies it. It also identifies gaps where the Phase 3 V&V scripts provide coverage not present in the existing unit/integration test suite.

---

## 1. Existing Unit and Integration Test Coverage

### 1.1 Variogram (`variogram.py`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| Variogram validates `sill > 0` | [`test_variogram_v2_integration.py`](../../test_variogram_v2_integration.py) | `test_variogram_integration` | ✅ Covered |
| Variogram loads from config file and exposes `sill`, `range_`, `nugget` | [`test_variogram_v2_integration.py`](../../test_variogram_v2_integration.py) | `test_variogram_integration` | ✅ Covered |
| Variogram integrates with `compute_resc` and `compute_polynomial_drift` | [`test_variogram_v2_integration.py`](../../test_variogram_v2_integration.py) | `test_variogram_integration` | ✅ Covered |

### 1.2 Coordinate Transformation (`transform.py`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| `apply_transform` → `invert_transform_coords` roundtrip recovers original coordinates | [`test_anisotropy_transformation.py`](../../test_anisotropy_transformation.py) | `test_coordinate_transformation_logic` | ✅ Covered |
| `get_transform_params` returns `center`, `R`, `S` keys | [`test_anisotropy_transformation.py`](../../test_anisotropy_transformation.py) | `test_coordinate_transformation_logic` | ✅ Covered |
| `center` is computed as the centroid of input points | [`test_anisotropy_transformation.py`](../../test_anisotropy_transformation.py) | `test_coordinate_transformation_logic` | ✅ Covered |
| `angle_major=0` (azimuth North) maps the North direction (+Y) onto the X-axis | [`test_anisotropy_transformation.py`](../../test_anisotropy_transformation.py) | `test_azimuth_convention_major_axis_alignment` | ✅ Covered |
| Azimuth convention: `angle_major` is CW from North; internally converted to arithmetic (CCW from East) | [`test_anisotropy_transformation.py`](../../test_anisotropy_transformation.py) | `test_azimuth_convention_major_axis_alignment` | ✅ Covered |

### 1.3 Rescaling Factor (`drift.py` — `compute_resc`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| `compute_resc` returns `sqrt(sill / radsqd)` in the normal case | [`test_drift.py`](../../test_drift.py) | `test_compute_resc_standard` | ✅ Covered |
| Safety floor activates when `radsqd < range²` (small domain relative to range) | [`test_drift.py`](../../test_drift.py) | `test_compute_resc_safety_floor` | ✅ Covered |
| Safety floor activates for a single point (radsqd = 0) | [`test_drift.py`](../../test_drift.py) | `test_compute_resc_single_point` | ✅ Covered |
| Safety floor does NOT activate when domain extent > range | [`test_drift.py`](../../test_drift.py) | `test_compute_resc_small_extent` | ✅ Covered |
| `compute_resc` returns a value < 1 for large domains | [`test_drift.py`](../../test_drift.py) | `test_compute_resc_inversion` | ✅ Covered |
| Safety floor activates (duplicate test in anisotropy file) | [`test_anisotropy_transformation.py`](../../test_anisotropy_transformation.py) | `test_compute_resc_safety_floor` | ✅ Covered |
| `compute_resc` with known corner-point inputs produces exact expected value | [`test_drift.py`](../../test_drift.py) | `test_drift_known_values` | ✅ Covered |

### 1.4 Polynomial Drift (`drift.py` — `compute_polynomial_drift`, `compute_drift_at_points`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| Drift term ordering is deterministic regardless of config dict key order | [`test_drift.py`](../../test_drift.py) | `test_deterministic_ordering` | ✅ Covered |
| Training drift matrix and prediction drift matrix are identical for same points | [`test_drift.py`](../../test_drift.py) | `test_consistency_training_prediction` | ✅ Covered |
| Partial drift config (only some terms enabled) produces correct column count | [`test_drift.py`](../../test_drift.py) | `test_partial_drift_terms` | ✅ Covered |
| `compute_drift_at_points` filters to only the requested term names | [`test_drift.py`](../../test_drift.py) | `test_prediction_term_filtering` | ✅ Covered |
| Unknown term names in `compute_drift_at_points` produce empty matrix (not error) | [`test_drift.py`](../../test_drift.py) | `test_unknown_term_error` | ✅ Covered |
| Empty drift config produces zero-column matrix | [`test_drift.py`](../../test_drift.py) | `test_empty_drift` | ✅ Covered |
| `linear_x` column = `resc * x`; `quadratic_y` column = `resc * y²` | [`test_drift.py`](../../test_drift.py) | `test_drift_values` | ✅ Covered |
| Known-value integration: resc + drift matrix with corner points | [`test_drift.py`](../../test_drift.py) | `test_drift_known_values` | ✅ Covered |
| Drift integrates with variogram class (end-to-end resc + drift) | [`test_drift.py`](../../test_drift.py) | `test_drift_with_variogram_integration` | ✅ Covered |

### 1.5 Drift Diagnostics (`drift.py` — `drift_diagnostics`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| Single-term drift matrix does not log a correlation matrix | [`test_drift.py`](../../test_drift.py) | `test_diagnostics_single_term` | ✅ Covered |
| Highly correlated drift columns trigger a "Drift Magnitude Check" log | [`test_drift.py`](../../test_drift.py) | `test_diagnostics_high_correlation` | ✅ Covered |
| Empty drift matrix logs "Drift Magnitude Check skipped" | [`test_drift.py`](../../test_drift.py) | `test_diagnostics_empty` | ✅ Covered |

### 1.6 Kriging Model (`kriging.py`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| `build_uk_model` with drift passes `drift_terms=['specified']` to PyKrige | [`test_kriging.py`](../../test_kriging.py) | `test_build_uk_model_with_drift` | ✅ Covered |
| `build_uk_model` with drift passes `specified_drift` list to PyKrige | [`test_kriging.py`](../../test_kriging.py) | `test_build_uk_model_with_drift` | ✅ Covered |
| `build_uk_model` with no drift does NOT pass `drift_terms` to PyKrige | [`test_kriging.py`](../../test_kriging.py) | `test_build_uk_model_no_drift` | ✅ Covered |
| `predict_at_points` raises `ValueError` when drift column count mismatches training | [`test_kriging.py`](../../test_kriging.py) | `test_predict_at_points_column_mismatch` | ✅ Covered |
| `output_drift_coefficients` returns OLS coefficient array of correct shape | [`test_kriging.py`](../../test_kriging.py) | `test_output_ols_coeffs` | ✅ Covered |
| `predict_on_grid` output arrays match meshgrid shape | [`test_kriging.py`](../../test_kriging.py) | `test_predict_on_grid_shapes` | ✅ Covered |
| `cross_validate` returns dict with `rmse`, `mae`, and `predictions` keys | [`test_kriging.py`](../../test_kriging.py) | `test_cross_validate_small` | ✅ Covered |
| PyKrige anisotropy is disabled (`anisotropy_scaling=1.0`, `anisotropy_angle=0.0`) after pre-transformation | [`test_anisotropy_transformation.py`](../../test_anisotropy_transformation.py) | `test_anisotropy_disabled_in_pykrige` | ✅ Covered |

### 1.7 End-to-End Integration (`test_kriging_integration.py`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| Full pipeline: resc → drift matrix → UK model → grid prediction → LOOCV → OLS | [`test_kriging_integration.py`](../../test_kriging_integration.py) | `test_kriging_integration_e2e` | ✅ Covered |
| Full pipeline with no drift (ordinary kriging path) | [`test_kriging_integration.py`](../../test_kriging_integration.py) | `test_kriging_integration_no_drift` | ✅ Covered |

### 1.8 Output Functions (`main.py`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| `export_contours` writes a shapefile with `elevation` column | [`test_main.py`](../../test_main.py) | `test_export_contours_valid` | ✅ Covered |
| `export_contours` raises `ValueError` for interval ≤ 0 | [`test_main.py`](../../test_main.py) | `test_export_contours_invalid_interval` | ✅ Covered |
| `export_contours` logs warning when Z grid is all NaN | [`test_main.py`](../../test_main.py) | `test_export_contours_no_levels` | ✅ Covered |
| `export_contours` creates output directory if it does not exist | [`test_main.py`](../../test_main.py) | `test_export_contours_creates_dir` | ✅ Covered |
| `export_aux_points` writes shapefile with `head` column and correct geometry | [`test_main.py`](../../test_main.py) | `test_export_aux_points_valid` | ✅ Covered |
| `export_aux_points` handles empty arrays gracefully | [`test_main.py`](../../test_main.py) | `test_export_aux_points_empty` | ✅ Covered |
| `generate_map` calls `savefig` with the configured output path | [`test_main.py`](../../test_main.py) | `test_generate_map_save` | ✅ Covered |
| `diagnose_kriging_system` logs exact interpolation test result | [`test_main.py`](../../test_main.py) | `test_diagnose_kriging_system_logging` | ✅ Covered |
| `diagnose_kriging_system` logs warning when drift magnitude ratio > 1000 | [`test_main.py`](../../test_main.py) | `test_diagnose_kriging_system_warning` | ✅ Covered |

### 1.9 Data Loading (`data.py`)

| Claim | Test File | Test Function | Status |
|---|---|---|---|
| `load_line_features` handles `LineString` geometry | [`test_data.py`](../../test_data.py) | Multiple tests | ✅ Covered |
| `load_line_features` handles `MultiLineString` geometry | [`test_data.py`](../../test_data.py) | Multiple tests | ✅ Covered |
| `load_line_features` handles missing stage columns | [`test_data.py`](../../test_data.py) | Multiple tests | ✅ Covered |
| `load_line_features` handles empty shapefile | [`test_data.py`](../../test_data.py) | Multiple tests | ✅ Covered |
| `load_observation_wells` loads point shapefile | [`test_data.py`](../../test_data.py) | Multiple tests | ✅ Covered |
| `prepare_data` merges observation wells and control points | [`test_data.py`](../../test_data.py) | Multiple tests | ✅ Covered |

---

## 2. Phase 3 V&V Script Coverage

The following table identifies behaviors verified by the Phase 3 V&V scripts that are **not** covered by the existing unit/integration test suite. These scripts provide analytical ground-truth validation (hand-calculated expected values, symmetry checks, superposition checks) that go beyond the functional correctness checks in the unit tests.

### 2.1 Variogram Model Equations (`vv_variogram_models.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| `γ(0) = 0` for all models (or nugget behavior at h→0) | [`vv_variogram_models.py`](vv_variogram_models.py) | TC-1 through TC-4 | ⚠️ **Gap** — unit tests do not verify semivariance values at specific h |
| `γ(range) ≈ sill` for each model within model-specific tolerance | [`vv_variogram_models.py`](vv_variogram_models.py) | TC-1 through TC-4 | ⚠️ **Gap** — no unit test checks semivariance at h=range |
| `γ(2*range) = sill` for bounded models (spherical, linear) | [`vv_variogram_models.py`](vv_variogram_models.py) | TC-1, TC-4 | ⚠️ **Gap** — no unit test checks sill plateau behavior |
| `γ(range/2)` matches hand-calculated analytical value | [`vv_variogram_models.py`](vv_variogram_models.py) | TC-1 through TC-4 | ⚠️ **Gap** — no unit test checks mid-range semivariance |
| `nugget > sill` raises `ValueError` | [`vv_variogram_models.py`](vv_variogram_models.py) | TC-5 | ⚠️ **Gap** — unit tests do not test invalid parameter validation |
| `range < 0` raises `ValueError` | [`vv_variogram_models.py`](vv_variogram_models.py) | TC-6 | ⚠️ **Gap** — unit tests do not test invalid parameter validation |

### 2.2 Coordinate Transformation Roundtrip (`vv_transform_roundtrip.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| Roundtrip error < 1e-12 for identity transform (angle=0°, ratio=1.0) | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | TC-1 | ⚠️ **Gap** — unit test uses `np.allclose` (1e-8 tolerance), not 1e-12 |
| Roundtrip error < 1e-12 for angle=45°, ratio=0.5 | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | TC-2 | ⚠️ **Gap** — unit test only tests angle=45° at 1e-8 tolerance |
| Roundtrip error < 1e-12 for angle=90°, ratio=0.3 | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | TC-3 | ⚠️ **Gap** — angle=90° not tested in unit tests |
| Roundtrip error < 1e-12 for pure scaling (angle=0°, ratio=0.5) | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | TC-4 | ⚠️ **Gap** — pure scaling case not tested in unit tests |
| Roundtrip works for a single point at origin | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | TC-5 | ⚠️ **Gap** — single-point edge case not in unit tests |
| Roundtrip works for collinear points | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | TC-6 | ⚠️ **Gap** — collinear edge case not in unit tests |
| After angle=0° azimuth (arithmetic 90°), ratio=0.5: Y-coords scaled by 2x | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | Check A | ⚠️ **Gap** — scaling magnitude not verified in unit tests |
| After angle=90° azimuth (arithmetic 0°), ratio=1.0: X and Y coords swapped | [`vv_transform_roundtrip.py`](vv_transform_roundtrip.py) | Check B | ⚠️ **Gap** — swap behavior not verified in unit tests |

### 2.3 Polynomial Drift Computation (`vv_polynomial_drift.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| `linear_x` column = `resc * x` with exact numerical values | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-1 | ✅ Also covered by `test_drift_values` |
| `linear_y` column = `resc * y` with exact numerical values | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-2 | ✅ Also covered by `test_drift_values` |
| `quadratic_x` column = `resc * x²` with exact numerical values | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-3 | ✅ Also covered by `test_drift_values` |
| `quadratic_y` column = `resc * y²` with exact numerical values | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-4 | ✅ Also covered by `test_drift_values` |
| Term ordering is `[linear_x, linear_y, quadratic_x, quadratic_y]` for all 4 terms | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-5 | ⚠️ **Gap** — unit tests only test subsets; all-4-terms ordering not verified |
| `compute_resc` normal case: `x=[0,100], y=[0,100], sill=1, range=50` → `resc=0.01` | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-6a | ✅ Also covered by `test_compute_resc_standard` |
| `compute_resc` safety floor: `x=[0,1], y=[0,1], sill=1, range=1000` → `resc=0.001` | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-6b | ✅ Also covered by `test_compute_resc_safety_floor` |
| Empty config produces zero-column matrix | [`vv_polynomial_drift.py`](vv_polynomial_drift.py) | TC-7 | ✅ Also covered by `test_empty_drift` |

### 2.4 Drift Physics Verification (`vv_drift_physics.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| `verify_drift_physics` returns PASS for valid `linear_x` column | [`vv_drift_physics.py`](vv_drift_physics.py) | TC-1 | ⚠️ **Gap** — `verify_drift_physics` not tested in unit tests |
| `verify_drift_physics` returns PASS for valid `quadratic_y` column | [`vv_drift_physics.py`](vv_drift_physics.py) | TC-2 | ⚠️ **Gap** — `verify_drift_physics` not tested in unit tests |
| `verify_drift_physics` returns FAIL for corrupted `linear_x` (R² < 0.999) | [`vv_drift_physics.py`](vv_drift_physics.py) | TC-3 | ⚠️ **Gap** — `verify_drift_physics` not tested in unit tests |
| `verify_drift_physics` returns FAIL for wrong scaling (slope error > 1%) | [`vv_drift_physics.py`](vv_drift_physics.py) | TC-4 | ⚠️ **Gap** — `verify_drift_physics` not tested in unit tests |
| `verify_drift_physics` returns SKIP for AEM term names (no `_x` or `_y`) | [`vv_drift_physics.py`](vv_drift_physics.py) | TC-5 | ⚠️ **Gap** — `verify_drift_physics` not tested in unit tests |

### 2.5 AEM Linesink Potential — Single Segment (`vv_aem_single_segment.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| Potential is symmetric about segment axis: `φ(x, y) = φ(x, -y)` | [`vv_aem_single_segment.py`](vv_aem_single_segment.py) | TC-2 | ⚠️ **Gap** — AEM potential not tested in unit tests at all |
| Superposition: two collinear half-segments equal one full segment | [`vv_aem_single_segment.py`](vv_aem_single_segment.py) | TC-3 | ⚠️ **Gap** — AEM superposition not tested in unit tests |
| Zero-length segment returns zeros (singularity guard) | [`vv_aem_single_segment.py`](vv_aem_single_segment.py) | TC-4 | ⚠️ **Gap** — AEM edge cases not tested in unit tests |
| Strength linearity: `φ(strength=2) = 2 * φ(strength=1)` | [`vv_aem_single_segment.py`](vv_aem_single_segment.py) | TC-5 | ⚠️ **Gap** — AEM linearity not tested in unit tests |

### 2.6 AEM Drift Scaling Consistency (`vv_aem_scaling_consistency.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| Scaling factors from training are reused exactly during prediction | [`vv_aem_scaling_consistency.py`](vv_aem_scaling_consistency.py) | TC-4 | ⚠️ **Gap** — AEM scaling factor persistence not tested in unit tests |
| Without `input_scaling_factors`, prediction scaling differs from training | [`vv_aem_scaling_consistency.py`](vv_aem_scaling_consistency.py) | TC-5 | ⚠️ **Gap** — AEM scaling factor contract not tested in unit tests |
| Fixed rescaling method: factor = `sill / 0.0001` regardless of data | [`vv_aem_scaling_consistency.py`](vv_aem_scaling_consistency.py) | TC-6 | ⚠️ **Gap** — fixed rescaling method not tested in unit tests |

### 2.7 Wrapper Equivalence — No Drift (`vv_wrapper_no_drift.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| `build_uk_model` with empty drift produces identical predictions to direct PyKrige (< 1e-10) | [`vv_wrapper_no_drift.py`](vv_wrapper_no_drift.py) | TC-1 | ⚠️ **Gap** — numerical equivalence to PyKrige not tested in unit tests |
| `build_uk_model` with empty drift produces identical variances to direct PyKrige (< 1e-10) | [`vv_wrapper_no_drift.py`](vv_wrapper_no_drift.py) | TC-1 | ⚠️ **Gap** — variance equivalence not tested in unit tests |

### 2.8 Polynomial Drift Recovery (`vv_polynomial_drift_recovery.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| UK with linear drift recovers known trend `0.5*x + 0.3*y` within noise bounds | [`vv_polynomial_drift_recovery.py`](vv_polynomial_drift_recovery.py) | TC-1 | ⚠️ **Gap** — trend recovery accuracy not tested in unit tests |
| Our wrapper matches direct PyKrige specified-drift predictions (< 1e-8) | [`vv_polynomial_drift_recovery.py`](vv_polynomial_drift_recovery.py) | TC-2 | ⚠️ **Gap** — numerical equivalence with specified drift not tested in unit tests |

### 2.9 Anisotropy Pre-Transform Consistency (`vv_anisotropy_consistency.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| Pre-transform approach produces equivalent predictions to PyKrige internal anisotropy (< 1e-6) | [`vv_anisotropy_consistency.py`](vv_anisotropy_consistency.py) | TC-1 | ⚠️ **Gap** — numerical equivalence to PyKrige anisotropy not tested in unit tests |
| After transform with azimuth 90° (arithmetic 0°), ratio=0.5: Y-coords scaled by 2x | [`vv_anisotropy_consistency.py`](vv_anisotropy_consistency.py) | TC-3 | ⚠️ **Gap** — scaling magnitude not verified in unit tests |
| Equivalence holds at angle=0°, 45°, 90° with ratio=0.5 | [`vv_anisotropy_consistency.py`](vv_anisotropy_consistency.py) | TC-1 through TC-3 | ⚠️ **Gap** — multi-angle equivalence not tested in unit tests |

### 2.10 LOOCV Diagnostic Metrics (`vv_loocv.py`)

| Claim | V&V Script | Test Case | Gap vs Unit Tests |
|---|---|---|---|
| `cross_validate` produces exactly N predictions for N input points | [`vv_loocv.py`](vv_loocv.py) | TC-1 | ⚠️ **Gap** — prediction count not verified in unit tests |
| Each LOOCV prediction excludes the held-out point | [`vv_loocv.py`](vv_loocv.py) | TC-2 | ⚠️ **Gap** — hold-out exclusion not verified in unit tests |
| RMSE = `sqrt(mean((pred - obs)²))` computed correctly | [`vv_loocv.py`](vv_loocv.py) | TC-3 | ⚠️ **Gap** — RMSE formula not verified in unit tests |
| MAE = `mean(|pred - obs|)` computed correctly | [`vv_loocv.py`](vv_loocv.py) | TC-4 | ⚠️ **Gap** — MAE formula not verified in unit tests |
| Q1 and Q2 (standardized error statistics) are computed | [`vv_loocv.py`](vv_loocv.py) | TC-5 | ⚠️ **Gap** — Q1/Q2 not verified in unit tests |
| With < 3 points, function returns NaN metrics gracefully | [`vv_loocv.py`](vv_loocv.py) | TC-6 | ⚠️ **Gap** — small-dataset edge case not tested in unit tests |

---

## 3. Coverage Gap Summary

| Module | Unit Test Coverage | V&V Additional Coverage |
|---|---|---|
| `variogram.py` — model equations | ❌ Not tested | ✅ `vv_variogram_models.py` — all 4 models, analytical values, validation errors |
| `variogram.py` — parameter validation | ❌ Not tested | ✅ `vv_variogram_models.py` — nugget > sill, range < 0 |
| `transform.py` — roundtrip precision | ⚠️ 1e-8 tolerance | ✅ `vv_transform_roundtrip.py` — 1e-12 tolerance, 6 cases |
| `transform.py` — scaling magnitude | ❌ Not tested | ✅ `vv_transform_roundtrip.py` — Y scaled by 1/ratio |
| `drift.py` — all-4-terms ordering | ⚠️ Partial (subsets only) | ✅ `vv_polynomial_drift.py` — all 4 terms in all orderings |
| `drift.py` — `verify_drift_physics` | ❌ Not tested | ✅ `vv_drift_physics.py` — PASS/FAIL/SKIP cases |
| `AEM_drift.py` — potential values | ❌ Not tested | ✅ `vv_aem_single_segment.py` — symmetry, superposition, linearity |
| `AEM_drift.py` — scaling persistence | ❌ Not tested | ✅ `vv_aem_scaling_consistency.py` — training/prediction factor reuse |
| `kriging.py` — PyKrige equivalence | ❌ Not tested | ✅ `vv_wrapper_no_drift.py` — numerical equivalence < 1e-10 |
| `kriging.py` — trend recovery | ❌ Not tested | ✅ `vv_polynomial_drift_recovery.py` — RMSE vs true trend |
| `kriging.py` — anisotropy equivalence | ❌ Not tested | ✅ `vv_anisotropy_consistency.py` — pre-transform vs PyKrige internal |
| `kriging.py` — LOOCV metric formulas | ⚠️ Keys present only | ✅ `vv_loocv.py` — RMSE, MAE, Q1, Q2 formulas verified |

---

## 4. Recommended Unit Test Additions

The following behaviors are currently only covered by V&V scripts and would benefit from promotion to the unit test suite for regression protection:

1. **Variogram model equations** — Add `test_variogram_semivariance_values.py` testing `γ(0)`, `γ(range)`, `γ(2*range)` for all 4 models.
2. **Variogram parameter validation** — Add tests for `nugget > sill` and `range < 0` raising `ValueError`.
3. **`verify_drift_physics` PASS/FAIL/SKIP** — Add to `test_drift.py`.
4. **AEM potential symmetry and linearity** — Add `test_aem_drift.py` with basic analytical checks.
5. **AEM scaling factor persistence** — Add test verifying `input_scaling_factors` are reused exactly.
6. **LOOCV metric formulas** — Add to `test_kriging.py` verifying RMSE and MAE calculations.
