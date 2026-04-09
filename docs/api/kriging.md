# API Reference: `kriging.py`

Provides the Universal Kriging model wrapper, grid and point prediction, OLS drift diagnostics, and Leave-One-Out Cross Validation (LOOCV).

---

## `build_uk_model(all_x, all_y, all_h, drift_matrix, variogram)`

**Source:** [`kriging.py:46`](../../kriging.py:46)

Constructs and initialises a `pykrige.uk.UniversalKriging` model. This is the central training function of the pipeline.

### PyKrige Wrapping Convention

When a non-empty drift matrix is provided, the model is initialised with:

```python
uk_kwargs["drift_terms"] = ["specified"]
uk_kwargs["specified_drift"] = [drift_matrix[:, i] for i in range(n_terms)]
```

This tells PyKrige to use externally supplied drift columns rather than its built-in polynomial drift. When `drift_matrix` is `None` or has zero columns, `drift_terms` is omitted entirely, producing an Ordinary Kriging model.

### Anisotropy Handling

The function reads `variogram.anisotropy_enabled` to decide how to configure PyKrige's internal anisotropy:

| `anisotropy_enabled` | PyKrige `anisotropy_scaling` | PyKrige `anisotropy_angle` | Meaning |
|---|---|---|---|
| `True` | `variogram.anisotropy_ratio` | `90.0 - variogram.angle_major` | PyKrige applies internal anisotropy (used when coordinates are **not** pre-transformed). |
| `False` | `1.0` | `0.0` | PyKrige anisotropy disabled (used when coordinates **are** pre-transformed by [`apply_transform()`](transform.md#apply_transform)). |

> **Important:** When the main pipeline pre-transforms coordinates via [`apply_transform()`](transform.md#apply_transform), it passes a cloned variogram with `anisotropy_enabled=False` to this function. This prevents double-application of the anisotropy transformation.

### Variogram Parameter Extraction

The function reads the following attributes from the `variogram` object:

| Attribute | Fallback | PyKrige key |
|---|---|---|
| `variogram.sill` | `0.0` | `variogram_parameters["sill"]` |
| `variogram.range_` or `variogram.range` | `0.0` | `variogram_parameters["range"]` |
| `variogram.nugget` | `0.0` | `variogram_parameters["nugget"]` |
| `variogram.model` | `"spherical"` | `variogram_model` |

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `all_x` | `np.ndarray` | 1-D array of training X coordinates (model space if anisotropy pre-transformed). |
| `all_y` | `np.ndarray` | 1-D array of training Y coordinates (model space if anisotropy pre-transformed). |
| `all_h` | `np.ndarray` | 1-D array of observed values (water levels). |
| `drift_matrix` | `np.ndarray` or `None` | Drift matrix of shape `(N, n_terms)`, or `None` / zero-column array for Ordinary Kriging. |
| `variogram` | `object` | Variogram-like object with `sill`, `range_`, `nugget`, `model`, and optional anisotropy attributes. |

### Returns

| Type | Description |
|---|---|
| `UniversalKriging` | Initialised PyKrige `UniversalKriging` instance ready for prediction. |

### Raises

| Exception | Condition |
|---|---|
| `ImportError` | `pykrige` is not installed. |
| Any exception from PyKrige | Re-raised after logging context. |

### Example

```python
import numpy as np
from variogram import variogram as Variogram
from drift import compute_resc, compute_polynomial_drift
from kriging import build_uk_model

x = np.array([0., 50., 100., 75., 25.])
y = np.array([0., 50., 0., 100., 75.])
h = np.array([10., 12., 11., 13., 11.5])

vario = Variogram({"variogram": {"model": "spherical", "sill": 1.0, "range": 80.0, "nugget": 0.1}})
resc = compute_resc(vario.sill, x, y, vario.range_)
drift_matrix, term_names = compute_polynomial_drift(x, y,
    {"drift_terms": {"linear_x": True}}, resc)

uk_model = build_uk_model(x, y, h, drift_matrix, vario)
```

---

## `predict_at_points(uk_model, x, y, drift_matrix_pred=None)`

**Source:** [`kriging.py:138`](../../kriging.py:138)

Predicts values and kriging variances at arbitrary point locations using a trained `UniversalKriging` model.

### Drift Column Count Validation

Before calling `uk_model.execute()`, the function validates that the number of drift columns at prediction time matches the number used during training:

```
n_train_cols  ← inferred from uk_model.specified_drift (or related attributes)
n_pred_cols   ← drift_matrix_pred.shape[1]

if n_train_cols != n_pred_cols:
    raise ValueError("Drift column count mismatch: ...")
```

This guards against the most common pipeline error: changing `term_names` between training and prediction.

### PyKrige Version Compatibility

PyKrige has changed the keyword argument name for specified drift across versions. The function tries the modern name first, then falls back:

```python
# Modern (preferred)
uk_model.execute("points", x, y, specified_drift_arrays=specified_drift_arrays)

# Fallback for older versions
uk_model.execute("points", x, y, specified_drift=specified_drift_arrays)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `uk_model` | `UniversalKriging` | Trained PyKrige model (or test double). |
| `x` | `array-like` | 1-D array of prediction X coordinates (model space). |
| `y` | `array-like` | 1-D array of prediction Y coordinates (model space). |
| `drift_matrix_pred` | `np.ndarray` or `None` | Drift matrix for prediction points, shape `(N_pred, n_terms)`. Required if model was trained with drift; ignored (with a warning) if model has no drift. |

### Returns

| Type | Description |
|---|---|
| `(np.ndarray, np.ndarray)` | Tuple `(z, ss)`: predicted values and kriging variances, both shape `(N_pred,)`. |

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `x` and `y` have different lengths. |
| `ValueError` | Model trained with drift but `drift_matrix_pred` is `None`. |
| `ValueError` | `drift_matrix_pred` is not 2-D when drift is expected. |
| `ValueError` | Drift column count mismatch between training and prediction. |

---

## `predict_on_grid(uk_model, config, term_names, resc, transform_params=None, scaling_factors=None)`

**Source:** [`kriging.py:260`](../../kriging.py:260)

Predicts on a regular grid defined by `config["grid"]`. Handles coordinate transformation, drift reconstruction (both polynomial and AEM), and calls [`predict_at_points()`](#predict_at_points).

### Grid Generation

Grid axes are built with `np.arange`:

```python
grid_x = np.arange(x_min, x_max, resolution)   # endpoint exclusive
grid_y = np.arange(y_min, y_max, resolution)
GX, GY = np.meshgrid(grid_x, grid_y)
```

The grid is defined in **raw space** (matching the input shapefile CRS). Grid coordinates are then transformed to model space if `transform_params` is provided.

### Coordinate Transformation

```python
if transform_params is not None:
    flat_x_model, flat_y_model = apply_transform(flat_x, flat_y, transform_params)
else:
    flat_x_model, flat_y_model = flat_x, flat_y
```

Prediction is always performed in model space (the space in which the kriging model was trained).

### Drift Reconstruction Logic

The function reconstructs the drift matrix column-by-column, preserving the exact `term_names` order from training:

| Term type | Identified by | Coordinates used | Function called |
|---|---|---|---|
| Polynomial | Name in `["linear_x", "linear_y", "quadratic_x", "quadratic_y"]` | Model space (`flat_x_model`, `flat_y_model`) | [`compute_drift_at_points()`](drift.md#compute_drift_at_points) |
| AEM linesink | Any other name (group column value) | Model space if `apply_anisotropy=True`; raw space if `False` | [`compute_linesink_drift_matrix()`](aem_drift.md#compute_linesink_drift_matrix) |

For AEM terms, the `apply_anisotropy` flag is read from `config["drift_terms"]["linesink_river"]["apply_anisotropy"]`. When `False`, raw grid coordinates (`flat_x`, `flat_y`) are passed to the AEM function instead of model-space coordinates.

### `scaling_factors` Parameter

The `scaling_factors` dict (returned by [`compute_linesink_drift_matrix()`](aem_drift.md#compute_linesink_drift_matrix) during training) is passed as `input_scaling_factors` to each AEM column reconstruction call. This ensures the same normalisation is applied at prediction time as at training time.

```python
col_data, _, _ = compute_linesink_drift_matrix(
    ...,
    input_scaling_factors=scaling_factors  # from training
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `uk_model` | `UniversalKriging` | Trained PyKrige model. |
| `config` | `dict` | Full configuration dict containing `"grid"` sub-dict with `x_min`, `x_max`, `y_min`, `y_max`, `resolution`. Also used to read `data_sources.linesink_river` and `drift_terms` if AEM terms are present. |
| `term_names` | `list[str]` | Ordered list of drift term names from training. Must be identical to those used in `build_uk_model()`. |
| `resc` | `float` | Rescaling factor from training. Must be the same value used during training. |
| `transform_params` | `dict` or `None` | Anisotropy transform parameters from [`get_transform_params()`](transform.md#get_transform_params). `None` if no anisotropy. |
| `scaling_factors` | `dict` or `None` | AEM scaling factors from training. `None` if no AEM drift terms. |

### Returns

| Type | Description |
|---|---|
| `(GX, GY, Z_grid, SS_grid)` | 4-tuple of `np.ndarray`. `GX` and `GY` are meshgrid arrays in raw space. `Z_grid` and `SS_grid` are prediction and variance grids with the same shape as `GX`. |

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `config` is not a dict. |
| `ValueError` | `config["grid"]` is missing or not a dict. |
| `ValueError` | Grid parameters are non-numeric or missing. |
| `ValueError` | `resolution ≤ 0`. |
| `ValueError` | `x_min ≥ x_max` or `y_min ≥ y_max`. |
| `ValueError` | Grid axes are empty after applying resolution. |

---

## `output_drift_coefficients(all_h, drift_matrix, term_names)`

**Source:** [`kriging.py:436`](../../kriging.py:436)

Computes OLS (Ordinary Least Squares) estimates of drift coefficients for diagnostic purposes. **Non-intrusive:** results are logged and returned but do not affect the kriging model state.

### Algorithm

Solves the least-squares problem:

```
drift_matrix · coeffs ≈ all_h
```

using `np.linalg.lstsq`. Each coefficient is logged with its corresponding term name.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `all_h` | `array-like` | Observed values, length `N`. |
| `drift_matrix` | `np.ndarray` or `None` | Design matrix of shape `(N, p)`. If `None` or zero-column, returns `None`. |
| `term_names` | `list[str]` | Term names corresponding to columns of `drift_matrix`. |

### Returns

| Type | Description |
|---|---|
| `np.ndarray` or `None` | Coefficient vector of length `p` if solvable; `None` if `drift_matrix` is `None`, zero-column, or if `lstsq` fails. |

### Notes

- This is a diagnostic tool only. The OLS coefficients are not used by the kriging model.
- Useful for sanity-checking that drift terms have physically meaningful magnitudes relative to the observed values.
- Failures (e.g., rank-deficient matrix) are caught and logged as warnings; `None` is returned.

---

## `cross_validate(all_x, all_y, all_h, config, variogram)`

**Source:** [`kriging.py:489`](../../kriging.py:489)

Leave-One-Out Cross Validation (LOOCV) for Universal Kriging. For each point `i`, trains a model on the remaining `n-1` points and predicts at point `i`.

### LOOCV Algorithm

For each fold `i = 0 … n-1`:

1. **Exclude** point `i` from the training set.
2. **Recompute** `resc` on the `n-1` training points via [`compute_resc()`](drift.md#compute_resc).
3. **Recompute** polynomial drift matrix for the `n-1` training points via [`compute_polynomial_drift()`](drift.md#compute_polynomial_drift).
4. **Retrain** a `UniversalKriging` model on the `n-1` points via [`build_uk_model()`](#build_uk_model).
5. **Reconstruct** drift at the held-out point using the same `term_names` and `resc` from step 2–3.
6. **Predict** at the held-out point via [`predict_at_points()`](#predict_at_points).

> **Note:** `resc` is recomputed per fold on the training subset. This is the statistically correct approach — the held-out point must not influence the rescaling factor.

### Minimum Data Requirement

If `n < 3`, the function returns immediately with all metrics set to `float("nan")` and empty arrays. This prevents degenerate kriging systems.

### Error Statistics

Computed over all successfully predicted folds (folds where prediction did not raise an exception):

| Metric | Formula | Description |
|---|---|---|
| `rmse` | `sqrt(mean((pred - obs)²))` | Root Mean Squared Error |
| `mae` | `mean(|pred - obs|)` | Mean Absolute Error |
| `q1` | `nanmean(errors / sqrt(variances))` | Mean standardised error (should be ≈ 0 for unbiased model) |
| `q2` | `nanvar(errors / sqrt(variances))` | Variance of standardised errors (should be ≈ 1 for correct variance model) |

Standardised errors use `float("nan")` when the predicted variance is `≤ 1e-12` or `NaN`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `all_x` | `np.ndarray` | 1-D array of X coordinates. |
| `all_y` | `np.ndarray` | 1-D array of Y coordinates. |
| `all_h` | `np.ndarray` | 1-D array of observed values. |
| `config` | `dict` | Configuration dict (same format as main pipeline). Used to determine which drift terms to include in each fold. |
| `variogram` | `object` | Variogram object with `sill`, `range_`, `nugget`, `model` attributes. |

### Returns

`dict` with the following keys:

| Key | Type | Description |
|---|---|---|
| `"rmse"` | `float` | Root Mean Squared Error (or `nan` if no valid predictions). |
| `"mae"` | `float` | Mean Absolute Error (or `nan`). |
| `"q1"` | `float` | Mean standardised error (or `nan`). |
| `"q2"` | `float` | Variance of standardised errors (or `nan`). |
| `"predictions"` | `np.ndarray` | Per-point predictions, shape `(n,)`. Failed folds contain `nan`. |
| `"variances"` | `np.ndarray` | Per-point kriging variances, shape `(n,)`. Failed folds contain `nan`. |
| `"observations"` | `np.ndarray` | Original observed values `all_h`, shape `(n,)`. |

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `all_x`, `all_y`, `all_h` have different lengths. |

### Example

```python
import numpy as np
from variogram import variogram as Variogram
from kriging import cross_validate

x = np.random.default_rng(42).uniform(0, 100, 20)
y = np.random.default_rng(43).uniform(0, 100, 20)
h = 0.5 * x + np.random.default_rng(44).normal(0, 2, 20)

config = {
    "variogram": {"model": "spherical", "sill": 100.0, "range": 60.0, "nugget": 4.0},
    "drift_terms": {"linear_x": True, "linear_y": False,
                    "quadratic_x": False, "quadratic_y": False},
}
vario = Variogram(config)
results = cross_validate(x, y, h, config, vario)

print(f"RMSE: {results['rmse']:.4f}")
print(f"MAE:  {results['mae']:.4f}")
print(f"Q1:   {results['q1']:.4f}")   # ≈ 0 → unbiased
print(f"Q2:   {results['q2']:.4f}")   # ≈ 1 → correct variance
```

---

## Module Exports

```python
__all__ = [
    "build_uk_model",
    "predict_at_points",
    "predict_on_grid",
    "output_drift_coefficients",
    "cross_validate",
]
```

---

## Key Contracts Summary

| Contract | Where Enforced |
|---|---|
| `term_names` order must be identical between training and prediction | [`predict_at_points()`](#predict_at_points) column count check |
| AEM `scaling_factors` from training must be passed to `predict_on_grid()` | [`predict_on_grid()`](#predict_on_grid) `scaling_factors` parameter |
| Variogram must have `anisotropy_enabled=False` when coordinates are pre-transformed | [`build_uk_model()`](#build_uk_model) anisotropy block |
| `resc` must be the same value at training and prediction | Caller responsibility; enforced by convention |

---

## See Also

- [`docs/workflow.md`](../workflow.md) — Full pipeline execution order showing where each function is called.
- [`docs/api/drift.md`](drift.md) — Polynomial drift computation functions consumed by this module.
- [`docs/api/aem_drift.md`](aem_drift.md) — AEM drift matrix construction consumed by `predict_on_grid()`.
- [`docs/api/transform.md`](transform.md) — Coordinate transformation applied before training and prediction.
- [`docs/theory/anisotropy.md`](../theory/anisotropy.md) — Why PyKrige anisotropy is disabled after pre-transformation.
