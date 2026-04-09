# Polynomial Drift Theory

**File:** `docs/theory/polynomial-drift.md`  
**Task:** 2.3  
**Source:** [`drift.py`](../drift.py)

---

## 1. Universal Kriging Formulation

Universal Kriging (UK) decomposes the spatial field into a deterministic **drift** (trend) and a stochastic **residual**:

```
Z(x) = m(x) + ε(x)
```

| Component | Symbol | Meaning |
|---|---|---|
| Observed value | `Z(x)` | Measured water level at location `x` |
| Drift (trend) | `m(x)` | Deterministic large-scale trend |
| Residual | `ε(x)` | Zero-mean stationary random field |

The drift is modelled as a linear combination of known basis functions (drift terms):

```
m(x) = Σ βₖ · fₖ(x)
```

where `βₖ` are unknown coefficients estimated jointly with the kriging weights, and `fₖ(x)` are the drift basis functions.

This implementation uses the **specified drift** approach: the drift columns are computed externally and passed to PyKrige via `drift_terms="specified"`. This gives full control over the drift basis and allows mixing polynomial and AEM-based terms.

---

## 2. Supported Polynomial Drift Terms

Four polynomial basis functions are supported. All are computed in **model space** (after anisotropy transformation, if enabled).

| Config Key | Formula | Description |
|---|---|---|
| `linear_x` | `D = resc · x` | Linear trend along the X-axis |
| `linear_y` | `D = resc · y` | Linear trend along the Y-axis |
| `quadratic_x` | `D = resc · x²` | Quadratic (parabolic) trend along X |
| `quadratic_y` | `D = resc · y²` | Quadratic (parabolic) trend along Y |

Where `resc` is the rescaling factor (see Section 3) and `x`, `y` are model-space coordinates.

### Enabling Terms

Terms are enabled in `config.json` under `drift_terms`:

```json
"drift_terms": {
    "linear_x": true,
    "linear_y": true,
    "quadratic_x": false,
    "quadratic_y": false
}
```

Setting a key to `true` includes that term; `false` or omitting it excludes it.

---

## 3. The Rescaling Factor (`resc`)

### Formula

The rescaling factor is computed by [`compute_resc()`](../drift.py):

```
radsqd    = max( (xᵢ - x̄)² + (yᵢ - ȳ)² )   [max squared distance from centroid]
safe_radsqd = max(radsqd, range²)              [safety floor]
resc      = sqrt(sill / safe_radsqd)
```

In code:

```python
x_center = np.mean(x)
y_center = np.mean(y)
radsqd = np.max((x - x_center)**2 + (y - y_center)**2)
safe_radsqd = max(radsqd, variogram_range**2)
resc = np.sqrt(covmax / safe_radsqd)
```

where `covmax` is the variogram sill.

### Why Rescaling Is Necessary

Without rescaling, polynomial drift columns can have values orders of magnitude larger than the variogram sill. This causes the kriging system matrix to be poorly conditioned (near-singular), leading to numerical instability or nonsensical kriging weights.

The rescaling factor normalises the drift columns so that their maximum magnitude is approximately `sqrt(sill)`, making them numerically comparable to the covariance values in the kriging matrix.

**Intuition:** If the data spans a radius of `R` from the centroid, then `linear_x` has maximum value `resc · R = sqrt(sill / R²) · R = sqrt(sill)`. The drift column is thus scaled to the same order as the variogram sill.

### The Safety Floor

The safety floor `safe_radsqd = max(radsqd, range²)` prevents a specific failure mode:

> **Problem:** When the data extent is small relative to the variogram range (e.g., all points clustered within a small area), `radsqd` is tiny. This makes `resc` very large, causing the drift columns to dominate the kriging system.

> **Fix:** Floor `radsqd` at `range²`. This caps `resc` at `sqrt(sill) / range`, which is the natural scale of the variogram.

**Example:**
- Data: `x=[0, 1], y=[0, 1]`, sill=1.0, range=1000
- `radsqd = 0.5` (tiny)
- Without floor: `resc = sqrt(1/0.5) = 1.414` (too large)
- With floor: `safe_radsqd = max(0.5, 1000²) = 1,000,000`, `resc = sqrt(1/1,000,000) = 0.001` ✓

---

## 4. Term Ordering Contract

**The output column order is always fixed, regardless of the order keys appear in the config dict:**

```
[linear_x, linear_y, quadratic_x, quadratic_y]
```

This is enforced in [`compute_polynomial_drift()`](../drift.py) by iterating over a fixed list:

```python
for term in ["linear_x", "linear_y", "quadratic_x", "quadratic_y"]:
    if drift_cfg.get(term):
        term_names_to_compute.append(term)
```

**Why this matters:** The `term_names` list is used to reconstruct drift columns at prediction time (grid nodes). If the order changed between training and prediction, the drift coefficients `βₖ` would be applied to the wrong columns, producing incorrect predictions.

The `term_names` list returned by [`compute_polynomial_drift()`](../drift.py) must be stored and passed unchanged to [`compute_drift_at_points()`](../drift.py) during prediction.

---

## 5. Drift Computation at Prediction Points

At prediction time (grid nodes), drift columns are reconstructed using [`compute_drift_at_points()`](../drift.py):

```python
drift_matrix, computed_names = compute_drift_at_points(x_grid, y_grid, term_names, resc)
```

- `term_names` — the list returned from training (preserves order)
- `resc` — the **same** rescaling factor computed during training (not recomputed from grid points)
- `x_grid, y_grid` — model-space coordinates of grid nodes

> **Critical:** `resc` must not be recomputed from the grid coordinates. It must be the value computed from the training data, because the kriging coefficients `βₖ` were estimated assuming that specific scaling.

---

## 6. Drift Verification System

### `drift_diagnostics()`

[`drift_diagnostics()`](../drift.py) performs a magnitude check on each drift column relative to the variogram sill:

```
ratio = max(|drift_column|) / sill
```

A warning is logged if `ratio > 1000`, indicating the drift terms are disproportionately large and may destabilise the kriging system.

### `verify_drift_physics()`

[`verify_drift_physics()`](../drift.py) mathematically verifies that each drift column follows its theoretical equation. It is called after training and logs PASS/FAIL for each term.

**For linear terms** (`linear_x`, `linear_y`):

A degree-1 polynomial is fit to `drift_col` vs. the independent variable (`x` or `y`). Two checks are applied:

| Check | Criterion | Meaning |
|---|---|---|
| Shape (R²) | R² > 0.999 | Column is a perfect linear function |
| Scaling (slope) | `|slope - resc| / resc < 0.01` | Slope matches `resc` within 1% |

**For quadratic terms** (`quadratic_x`, `quadratic_y`):

A degree-2 polynomial is fit. Two checks are applied:

| Check | Criterion | Meaning |
|---|---|---|
| Shape (R²) | R² > 0.999 | Column is a perfect quadratic function |
| Scaling (curvature) | `|A - resc| / resc < 0.01` | Leading coefficient matches `resc` within 1% |

An additional **warning** (not a failure) is issued if the parabola vertex falls inside the data domain, which would cause the trend to reverse direction within the study area.

**For non-polynomial terms** (e.g., AEM linesink terms):

Terms whose names do not contain `_x` or `_y` are skipped with result `"SKIP"`.

**Return value:** A dict mapping each `term_name` to `"PASS"`, `"FAIL"`, `"SKIP"`, or `"ERROR"`.

---

## 7. Coordinate Space

All polynomial drift terms are computed in **model space** — i.e., after the anisotropy transformation (translate to centroid → rotate by `angle_major` → scale Y by `1/ratio`).

If anisotropy is disabled, model space equals raw space (no transformation is applied).

See [`docs/theory/anisotropy.md`](anisotropy.md) for the transformation details and [`docs/glossary.md`](../glossary.md) for the definition of raw vs. model space.

---

## 8. Summary of Key Contracts

| Contract | Where Enforced |
|---|---|
| Term order is always `[linear_x, linear_y, quadratic_x, quadratic_y]` | [`compute_polynomial_drift():52`](../drift.py) |
| `resc` is computed from training data, not grid | [`compute_drift_at_points()`](../drift.py) caller |
| `term_names` from training must be passed unchanged to prediction | [`compute_drift_at_points()`](../drift.py) |
| Drift columns are in model space (after anisotropy transform) | [`main.py`](../main.py) pipeline |
| Safety floor prevents instability for small data extents | [`compute_resc():23`](../drift.py) |
