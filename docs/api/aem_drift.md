# API Reference: `AEM_drift.py`

Provides Analytic Element Method (AEM) linesink potential computation and drift matrix construction for Universal Kriging with river-influence drift terms.

---

## `compute_linesink_potential(x, y, x1, y1, x2, y2, strength=1.0)`

**Source:** [`AEM_drift.py:7`](../../AEM_drift.py:7)

Computes the real-valued hydraulic potential at evaluation points `(x, y)` due to a single linesink segment from `(x1, y1)` to `(x2, y2)`. Based on the analytic element formulation from Strack (1989).

### Complex Potential Formula

The computation maps the physical segment to a normalised ZZ-space on `[-1, 1]`:

```
z        = x + i·y                          # evaluation point (complex)
z1       = x1 + i·y1                        # segment start (complex)
z2       = x2 + i·y2                        # segment end (complex)
mid      = (z1 + z2) / 2                    # segment midpoint
half_L   = (z2 - z1) / 2                    # half-length vector
ZZ       = (z - mid) / half_L               # normalised coordinate ∈ [-1, 1]

carg     = (ZZ+1)·ln(ZZ+1) - (ZZ-1)·ln(ZZ-1) + 2·ln(half_L) - 2

φ        = (strength · L / 4π) · Re(carg)  # real potential
```

where `L = |z2 - z1|` is the segment length.

### Singularity Handling

The logarithm `ln(ZZ ± 1)` is singular at the segment endpoints (`ZZ = +1` and `ZZ = -1`). The implementation perturbs `ZZ` by `±1e-10` when `|ZZ ∓ 1| < 1e-10` to avoid `NaN` or `Inf` values:

```python
mask     = np.abs(ZZ - 1.0) < 1e-10  →  ZZ[mask] += 1e-10
mask_neg = np.abs(ZZ + 1.0) < 1e-10  →  ZZ[mask_neg] = -1.0 - 1e-10
```

### Zero-Length Guard

If `L < 1e-6`, the function returns `np.zeros_like(x)` immediately without computation.

### Strength Scaling

The potential scales **linearly** with `strength`: `φ(strength=2) = 2 · φ(strength=1)` exactly (before singularity perturbation).

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` | 1-D array of X coordinates of evaluation points. |
| `y` | `np.ndarray` | 1-D array of Y coordinates of evaluation points. |
| `x1`, `y1` | `float` | Coordinates of the segment start point. |
| `x2`, `y2` | `float` | Coordinates of the segment end point. |
| `strength` | `float` | Linesink strength (hydraulic resistance). Default `1.0`. |

### Returns

| Type | Description |
|---|---|
| `np.ndarray` | Real-valued potential `φ` at each evaluation point, shape `(N,)`. |

### Coordinate Space

Evaluation points and segment endpoints must be in the **same coordinate space**. When `apply_anisotropy=True` in [`compute_linesink_drift_matrix()`](#compute_linesink_drift_matrix), both are in model space. When `apply_anisotropy=False`, both are in raw space.

### Reference

Strack, O. D. L. (1989). *Groundwater Mechanics*. Prentice Hall.

---

## `compute_linesink_drift_matrix(x_model, y_model, linesinks_gdf, group_col, transform_params, sill, strength_col='lval', rescaling_method='adaptive', apply_anisotropy=True, input_scaling_factors=None)`

**Source:** [`AEM_drift.py:53`](../../AEM_drift.py:53)

Constructs the AEM drift matrix for a set of evaluation points. Segments are grouped by `group_col`; each unique group value becomes one drift column (the sum of potentials from all segments in that group). Scaling factors are computed or reused to normalise each column relative to the variogram sill.

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `x_model` | `np.ndarray` | Yes | — | 1-D array of X coordinates of evaluation points. |
| `y_model` | `np.ndarray` | Yes | — | 1-D array of Y coordinates of evaluation points. |
| `linesinks_gdf` | `GeoDataFrame` | Yes | — | Shapefile data with linesink geometries and attributes. |
| `group_col` | `str` | Yes | — | Column name used to group segments into named linesinks (e.g., `"NAME"`). Each unique value becomes one drift column. |
| `transform_params` | `dict` or `None` | Yes | — | Transform parameter dict from [`get_transform_params()`](transform.md#get_transform_params). Used to transform segment endpoints to model space when `apply_anisotropy=True`. Pass `None` for no transformation. |
| `sill` | `float` | Yes | — | Variogram sill; used as the numerator in adaptive rescaling. |
| `strength_col` | `str` | No | `'lval'` | Column name in `linesinks_gdf` containing per-segment strength values. |
| `rescaling_method` | `str` | No | `'adaptive'` | `'adaptive'` or `'fixed'`. Controls how scaling factors are computed during training. See [Rescaling Methods](#rescaling-methods). |
| `apply_anisotropy` | `bool` | No | `True` | If `True`, segment endpoints are transformed to model space via `transform_params`. If `False`, raw coordinates are used for AEM computation. See [Coordinate Space Behaviour](#coordinate-space-behaviour). |
| `input_scaling_factors` | `dict` or `None` | No | `None` | Pre-computed scaling factors from a prior training call (keyed by group value). When provided, these factors are reused instead of recomputed. **Must be passed during prediction.** |

### Returns

| Type | Description |
|---|---|
| `(np.ndarray, list[str], dict)` | 3-tuple: `(drift_matrix, term_names, scaling_factors_used)`. |

| Return Element | Type | Description |
|---|---|---|
| `drift_matrix` | `np.ndarray` | Shape `(N_points, N_groups)`. Each column is the scaled potential for one linesink group. |
| `term_names` | `list[str]` | Ordered list of group identifiers (values of `group_col`), one per column. |
| `scaling_factors_used` | `dict` | Mapping of `group_value → rescaling_factor` applied to each column. Must be persisted and passed as `input_scaling_factors` during prediction. |

### Rescaling Methods

Each group's raw potential column is multiplied by a scaling factor `rescr` to normalise it relative to the variogram sill:

| Method | Formula | When Applied |
|---|---|---|
| **Adaptive** (default) | `rescr = sill / max(|φ|)` if `max(|φ|) > 1e-10`, else `rescr = 1.0` | Training phase (no `input_scaling_factors` provided) |
| **Fixed** (KT3D-style) | `rescr = sill / 0.0001` | When `rescaling_method='fixed'` |
| **Reuse** | `rescr = input_scaling_factors[group_id]` | Prediction phase (`input_scaling_factors` provided) |

Priority order: **Reuse > Fixed > Adaptive**.

### Coordinate Space Behaviour

The `apply_anisotropy` flag controls which coordinate space is used for AEM computation:

| `apply_anisotropy` | Segment endpoints | Evaluation points | Use case |
|---|---|---|---|
| `True` | Transformed to model space via `transform_params` | Model space (`x_model`, `y_model`) | River geometry participates in anisotropy transformation |
| `False` | Raw coordinates (no transformation applied) | Raw space (caller must pass raw coords as `x_model`, `y_model`) | KT3D-style: AEM computed in raw space, kriging in model space |

> **Note:** When `apply_anisotropy=False`, the caller in [`predict_on_grid()`](kriging.md#predict_on_grid) automatically switches to passing raw grid coordinates (`flat_x`, `flat_y`) instead of model-space coordinates.

### Critical Contract: Scaling Factor Persistence

Scaling factors computed during training **must** be reused during prediction. If they are not, the AEM drift columns at prediction points will be scaled differently from those at training points, corrupting the kriging system.

**Training phase:**
```python
drift_matrix, term_names, scaling_factors = compute_linesink_drift_matrix(
    x_train, y_train, linesinks_gdf, group_col,
    transform_params, sill,
    rescaling_method='adaptive'
    # input_scaling_factors=None  ← factors are computed here
)
# Persist scaling_factors for prediction
```

**Prediction phase:**
```python
drift_pred, _, _ = compute_linesink_drift_matrix(
    x_grid, y_grid, linesinks_gdf, group_col,
    transform_params, sill,
    rescaling_method='adaptive',
    input_scaling_factors=scaling_factors  # ← reuse training factors
)
```

In the full pipeline, `scaling_factors` is passed from training through [`predict_on_grid()`](kriging.md#predict_on_grid) via its `scaling_factors` parameter.

### Segment Iteration

For each group, the function iterates over all segments in the group and over all vertex pairs within each segment's geometry:

```
for each group_id:
    total_phi = 0
    for each segment in group:
        for each consecutive vertex pair (j, j+1):
            total_phi += compute_linesink_potential(x_model, y_model, x1, y1, x2, y2, strength)
    drift_matrix[:, i] = total_phi * rescr
```

### Example

```python
import geopandas as gpd
import numpy as np
from AEM_drift import compute_linesink_drift_matrix
from transform import get_transform_params

linesinks = gpd.read_file("rivers.shp")
x_train = np.array([100.0, 200.0, 300.0])
y_train = np.array([100.0, 150.0, 200.0])

transform_params = get_transform_params(x_train, y_train, angle=30.0, ratio=0.5)

# Training
drift_matrix, term_names, scaling_factors = compute_linesink_drift_matrix(
    x_train, y_train,
    linesinks, group_col="NAME",
    transform_params=transform_params,
    sill=1.5,
    strength_col="resistance",
    rescaling_method="adaptive",
    apply_anisotropy=True,
)

# Prediction — reuse scaling_factors
x_grid = np.linspace(50, 350, 100)
y_grid = np.linspace(50, 250, 100)
drift_pred, _, _ = compute_linesink_drift_matrix(
    x_grid, y_grid,
    linesinks, group_col="NAME",
    transform_params=transform_params,
    sill=1.5,
    strength_col="resistance",
    rescaling_method="adaptive",
    apply_anisotropy=True,
    input_scaling_factors=scaling_factors,
)
```

---

## Module Exports

`AEM_drift.py` does not define an explicit `__all__`. The public API consists of:

| Symbol | Description |
|---|---|
| [`compute_linesink_potential()`](#compute_linesink_potential) | Single-segment potential at evaluation points. |
| [`compute_linesink_drift_matrix()`](#compute_linesink_drift_matrix) | Full drift matrix for all linesink groups. |

---

## See Also

- [`docs/theory/aem-linesink.md`](../theory/aem-linesink.md) — Theory, complex potential derivation, and segment grouping rationale.
- [`docs/api/drift.md`](drift.md) — Polynomial drift; combined with AEM drift via `np.hstack` in the main pipeline.
- [`docs/api/kriging.md`](kriging.md) — `predict_on_grid()` which orchestrates AEM drift reconstruction at prediction time.
- [`docs/api/transform.md`](transform.md) — `apply_transform()` used internally to transform segment endpoints when `apply_anisotropy=True`.
