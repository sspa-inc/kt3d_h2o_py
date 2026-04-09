# API Reference: `drift.py`

Provides polynomial drift term computation, rescaling, diagnostics, and physics verification for Universal Kriging.

---

## `compute_resc(covmax, x, y, variogram_range)`

**Source:** [`drift.py:8`](../drift.py:8)

Computes the rescaling factor (`resc`) applied to all polynomial drift columns. The factor normalises drift magnitudes relative to the variogram sill and the spatial extent of the data, ensuring numerical stability of the kriging system matrix.

### Formula

```
radsqd     = max( (x - x̄)² + (y - ȳ)² )          # max squared distance from centroid
safe_radsqd = max(radsqd, variogram_range²)          # safety floor
resc        = sqrt(covmax / safe_radsqd)
```

If `safe_radsqd == 0` (degenerate single-point dataset), `resc` is set to `1.0`.

### Safety Floor

The floor `max(radsqd, variogram_range²)` prevents `resc` from becoming excessively large when the data domain is small relative to the correlation range. Without it, drift columns would dwarf the variogram covariance terms and destabilise the kriging matrix.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `covmax` | `float` | Variogram sill (maximum covariance). |
| `x` | `np.ndarray` | 1-D array of X coordinates (model space). |
| `y` | `np.ndarray` | 1-D array of Y coordinates (model space). |
| `variogram_range` | `float` | Variogram range; used as the safety floor for `radsqd`. |

### Returns

| Type | Description |
|---|---|
| `float` | Rescaling factor `resc ≥ 0`. |

### Example

```python
import numpy as np
from drift import compute_resc

x = np.array([0.0, 100.0])
y = np.array([0.0, 100.0])
resc = compute_resc(covmax=1.0, x=x, y=y, variogram_range=50.0)
# radsqd = 10000, safe_radsqd = max(10000, 2500) = 10000
# resc = sqrt(1.0 / 10000) = 0.01
```

---

## `compute_polynomial_drift(x, y, config, resc)`

**Source:** [`drift.py:34`](../drift.py:34)

Computes the polynomial drift matrix for a set of points. Supported terms are `linear_x`, `linear_y`, `quadratic_x`, and `quadratic_y`.

### Term Formulas

| Term | Column Formula |
|---|---|
| `linear_x` | `resc * x` |
| `linear_y` | `resc * y` |
| `quadratic_x` | `resc * x²` |
| `quadratic_y` | `resc * y²` |

### Term Ordering Contract

**The output column order is always `[linear_x, linear_y, quadratic_x, quadratic_y]`**, regardless of the order keys appear in the config dict. This canonical order is enforced by iterating over a fixed sequence:

```python
for term in ["linear_x", "linear_y", "quadratic_x", "quadratic_y"]:
    if drift_cfg.get(term):
        ...
```

This contract must be preserved between training and prediction. Any deviation will cause a drift column count mismatch error in [`predict_at_points()`](kriging.md#predict_at_points).

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` | 1-D array of X coordinates. |
| `y` | `np.ndarray` | 1-D array of Y coordinates. |
| `config` | `dict` or `list[str]` | If `dict`, reads `config["drift_terms"]` for enabled boolean flags. If `list`, treats the list as the ordered term names to compute directly (used by [`compute_drift_at_points()`](#compute_drift_at_points)). |
| `resc` | `float` | Rescaling factor from [`compute_resc()`](#compute_resc). |

### Returns

| Type | Description |
|---|---|
| `(np.ndarray, list[str])` | Tuple of `(drift_matrix, term_names)`. `drift_matrix` has shape `(N, n_terms)`. `term_names` is the ordered list of column identifiers. If no terms are enabled, returns `(zeros((N, 0)), [])`. |

### Example

```python
import numpy as np
from drift import compute_resc, compute_polynomial_drift

x = np.linspace(0, 100, 20)
y = np.linspace(0, 50, 20)
config = {"drift_terms": {"linear_x": True, "linear_y": True}}
resc = compute_resc(1.0, x, y, 50.0)

drift_matrix, term_names = compute_polynomial_drift(x, y, config, resc)
# term_names == ["linear_x", "linear_y"]
# drift_matrix.shape == (20, 2)
```

---

## `compute_drift_at_points(x, y, term_names, resc)`

**Source:** [`drift.py:83`](../drift.py:83)

Reconstructs polynomial drift columns at prediction-time points (e.g., grid nodes) using the same `term_names` and `resc` that were used during training. This is the prediction-phase counterpart to [`compute_polynomial_drift()`](#compute_polynomial_drift).

Internally delegates to [`compute_polynomial_drift()`](#compute_polynomial_drift) by passing `term_names` as the `config` argument (list form).

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` | 1-D array of X coordinates at prediction points (model space). |
| `y` | `np.ndarray` | 1-D array of Y coordinates at prediction points (model space). |
| `term_names` | `list[str]` | Ordered list of drift term names from training (e.g., `["linear_x", "quadratic_y"]`). Must match the training order exactly. |
| `resc` | `float` | Rescaling factor from training. Must be the same value used during training. |

### Returns

| Type | Description |
|---|---|
| `(np.ndarray, list[str])` | Tuple of `(drift_matrix, computed_names)`. `drift_matrix` has shape `(N_points, N_terms)`. `computed_names` mirrors `term_names`. |

### Critical Contract

`term_names` and `resc` must be identical to those used during training. Passing different values will produce a drift matrix that is inconsistent with the trained kriging model, leading to incorrect predictions or a column count mismatch error.

---

## `drift_diagnostics(drift_matrix, term_names, variogram=None)`

**Source:** [`drift.py:108`](../drift.py:108)

Performs magnitude diagnostics on the drift matrix and logs results. Does not modify any state; output is informational only.

### What It Checks

**Drift Magnitude Check:** For each drift column, computes:

```
ratio = max(|drift_column|) / sill
```

- Logs `ratio` for each term.
- Emits a `WARNING` log if `ratio > 1000`, indicating that drift terms are very large relative to the variogram sill. This can cause numerical instability in the kriging system.

If `variogram` is `None`, has no `sill` attribute, or `sill ≤ 0`, the magnitude check is skipped.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `drift_matrix` | `np.ndarray` | Drift matrix of shape `(N, n_terms)`. |
| `term_names` | `list[str]` | Ordered list of term names corresponding to columns. |
| `variogram` | `Any`, optional | Variogram object with a `sill` attribute. If `None`, magnitude check is skipped. |

### Returns

`None`. All output is via the `logging` module at `INFO` or `WARNING` level.

### Warning Threshold

| Condition | Log Level | Message |
|---|---|---|
| `ratio > 1000` | `WARNING` | `"Drift Magnitude Check WARNING: Term '...' ratio (...) to sill is very high (>1000)."` |

---

## `verify_drift_physics(drift_matrix, term_names, all_x, all_y, resc)`

**Source:** [`drift.py:262`](../drift.py:262)

Mathematically verifies that each drift column follows its theoretical equation. Returns a per-term PASS/FAIL/SKIP/ERROR result dict. This is a post-hoc sanity check, not a filter — it does not modify the drift matrix.

### Pass Criteria

**Linear terms** (`linear_x`, `linear_y`): Fits a degree-1 polynomial to `(indep_var, drift_col)` and checks:
1. R² > 0.999 (the column is a perfect linear function of the coordinate)
2. `|slope - resc| / resc < 0.01` (slope is within 1% of `resc`)

**Quadratic terms** (`quadratic_x`, `quadratic_y`): Fits a degree-2 polynomial and checks:
1. R² > 0.999 (the column is a perfect quadratic function of the coordinate)
2. `|A - resc| / resc < 0.01` (leading coefficient within 1% of `resc`)
3. *(Warning only, not a failure)* Vertex of the parabola is inside the data domain — this indicates a U-turn in the trend.

**AEM or unknown terms**: Any term name that does not contain `_x` or `_y` returns `"SKIP"`.

### Result Values

| Value | Meaning |
|---|---|
| `"PASS"` | All checks passed for this term. |
| `"FAIL"` | One or more checks failed (R² too low or slope/curvature error too large). |
| `"SKIP"` | Term name does not match a known polynomial pattern (e.g., AEM group names). |
| `"ERROR"` | An exception occurred during verification (e.g., degenerate input). |

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `drift_matrix` | `np.ndarray` | Drift matrix of shape `(N, n_terms)`. |
| `term_names` | `list[str]` | Ordered list of term names corresponding to columns. |
| `all_x` | `np.ndarray` | 1-D array of X coordinates used to compute the drift. |
| `all_y` | `np.ndarray` | 1-D array of Y coordinates used to compute the drift. |
| `resc` | `float` | Rescaling factor used when computing the drift. |

### Returns

| Type | Description |
|---|---|
| `dict[str, str]` | Mapping of `term_name → result_string`. Empty dict if `drift_matrix` is `None` or empty. |

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `len(term_names) != drift_matrix.shape[1]` |
| `ValueError` | `all_x.size != drift_matrix.shape[0]` |

### Example

```python
import numpy as np
from drift import compute_resc, compute_polynomial_drift, verify_drift_physics

x = np.linspace(0, 100, 30)
y = np.linspace(0, 50, 30)
config = {"drift_terms": {"linear_x": True, "quadratic_y": True}}
resc = compute_resc(1.0, x, y, 50.0)
drift_matrix, term_names = compute_polynomial_drift(x, y, config, resc)

results = verify_drift_physics(drift_matrix, term_names, x, y, resc)
# results == {"linear_x": "PASS", "quadratic_y": "PASS"}
```

---

## Internal Helpers

These functions are not exported in `__all__` but are called by [`verify_drift_physics()`](#verify_drift_physics):

| Function | Purpose |
|---|---|
| [`_verify_linear_term()`](../drift.py:143) | R² and slope check for linear drift columns. |
| [`_verify_quadratic_term()`](../drift.py:195) | R² and curvature check for quadratic drift columns; also warns if vertex is inside the data domain. |

---

## Module Exports

```python
__all__ = [
    "compute_resc",
    "compute_polynomial_drift",
    "compute_drift_at_points",
    "drift_diagnostics",
    "verify_drift_physics",
]
```

---

## See Also

- [`docs/theory/polynomial-drift.md`](../theory/polynomial-drift.md) — Theory and motivation for polynomial drift terms and the rescaling factor.
- [`docs/api/kriging.md`](kriging.md) — How drift matrices are consumed by `build_uk_model()` and `predict_on_grid()`.
- [`docs/api/aem_drift.md`](aem_drift.md) — AEM linesink drift, which is combined with polynomial drift via `np.hstack`.
