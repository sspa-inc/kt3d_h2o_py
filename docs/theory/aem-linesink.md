# AEM Linesink Drift Theory

**File:** `docs/theory/aem-linesink.md`  
**Task:** 2.4  
**Type:** DOC  
**Depends on:** [Glossary](../glossary.md)

---

## 1. The Analytic Element Method (AEM) Concept

The Analytic Element Method (AEM) is a technique from groundwater modelling in which the hydraulic influence of physical features — such as rivers, drains, and wells — is represented by superimposing analytical solutions to the governing flow equations. Each feature is an *analytic element* that contributes a known, closed-form potential field to the total hydraulic head.

In UK_SSPA v2, rivers are represented as **line sinks**: line segments that extract water from the aquifer (or inject it, depending on sign). Each line sink generates a **potential field** φ(x, y) that describes the hydraulic influence of that river segment at any point in the domain.

This potential field is used as a **drift term** in Universal Kriging. Rather than fitting a polynomial trend, the kriging model uses the physically-motivated AEM potential as the deterministic component of the mean structure:

```
Z(x) = β · φ(x) + ε(x)
```

where β is a fitted coefficient, φ(x) is the AEM potential, and ε(x) is the spatially-correlated residual.

**Reference:** Strack, O. D. L. (1989). *Groundwater Mechanics*. Prentice Hall.

---

## 2. The Complex Potential Formula

The potential for a single line sink segment is computed in [`compute_linesink_potential()`](../AEM_drift.py:7).

### 2.1 Coordinate Mapping to ZZ Space

The segment runs from endpoint **z₁ = x₁ + iy₁** to **z₂ = x₂ + iy₂** in the complex plane. The evaluation point is **z = x + iy**.

The segment is mapped to a normalised coordinate system ZZ ∈ [−1, +1] via:

```
mid      = (z₁ + z₂) / 2          (midpoint of segment)
half_L   = (z₂ − z₁) / 2          (half-length vector, complex)
ZZ       = (z − mid) / half_L      (normalised coordinate)
```

After this mapping:
- ZZ = −1 corresponds to endpoint z₁
- ZZ = 0 corresponds to the midpoint
- ZZ = +1 corresponds to endpoint z₂

```
                z₁ ──────────── mid ──────────── z₂
ZZ space:       −1               0               +1
                ↑                                 ↑
           ZZ = −1                           ZZ = +1
```

### 2.2 Complex Potential (carg)

The complex potential argument is:

```
carg = (ZZ + 1) · ln(ZZ + 1)  −  (ZZ − 1) · ln(ZZ − 1)  +  2 · ln(half_L)  −  2
```

This is the standard AEM line sink potential derived from the Cauchy integral representation of the stream function.

### 2.3 Real Potential φ

The physical potential (hydraulic head contribution) is the real part of `carg`, scaled by the segment strength and length:

```
φ(x, y) = (strength · L) / (4π) · Re(carg)
```

where:
- `strength` — the sink strength per unit length (from `strength_col` in the shapefile)
- `L = |z₂ − z₁|` — the Euclidean length of the segment
- `Re(·)` — the real part of the complex expression

### 2.4 Diagram: Single Linesink Segment

```
                        Evaluation point P = (x, y)
                              ·
                             /|
                            / |
                           /  |
                          /   |
                         /    |
                        /     |
    z₁ ────────────── mid ──────────── z₂
    (x₁,y₁)           ·              (x₂,y₂)
    ZZ = −1          ZZ = 0          ZZ = +1

    L = |z₂ − z₁|   (total segment length)
    half_L = (z₂ − z₁)/2   (complex half-length vector)
    ZZ = (z − mid) / half_L  (normalised coordinate of P)
```

---

## 3. Singularity Handling at Endpoints

The complex logarithm `ln(ZZ ± 1)` is singular when ZZ = ±1, i.e., when the evaluation point lies exactly on an endpoint of the segment.

The implementation in [`compute_linesink_potential()`](../AEM_drift.py:27) handles this by perturbing ZZ slightly away from the singular values:

```python
small = 1e-10

# Protect ZZ = +1 (end of segment)
mask = np.abs(ZZ - 1.0) < small
ZZ[mask] += small

# Protect ZZ = -1 (start of segment)
mask_neg = np.abs(ZZ + 1.0) < small
ZZ[mask_neg] = -1.0 - small
```

This ensures numerical stability when observation wells or control points coincide with segment endpoints. The perturbation is small enough (1e-10) that the resulting potential error is negligible.

**Zero-length segments:** If the segment length `L < 1e-6`, the function returns zeros immediately without computing the potential. This prevents division by zero in the ZZ mapping.

---

## 4. Segment Grouping

A river system typically consists of many individual line segments. In UK_SSPA v2, segments are grouped into named **linesinks** using the `group_column` attribute in the shapefile.

All segments sharing the same `group_column` value are treated as a single hydraulic feature. Their individual potentials are **summed** to produce the total potential for that group:

```
φ_group(x, y) = Σ φ_segment_k(x, y)   for all k in group
```

Each group becomes **one column** in the drift matrix. If there are N groups, the AEM drift matrix has N columns.

This grouping is performed in [`compute_linesink_drift_matrix()`](../AEM_drift.py:53):

```python
unique_ids = linesinks_gdf[group_col].unique()
drift_matrix = np.zeros((len(x_model), len(unique_ids)))

for i, linesink_id in enumerate(unique_ids):
    segments = linesinks_gdf[linesinks_gdf[group_col] == linesink_id]
    total_phi = np.zeros_like(x_model)
    for _, row in segments.iterrows():
        # ... accumulate potential from each segment
        total_phi += compute_linesink_potential(...)
    drift_matrix[:, i] = total_phi * rescr
```

The `term_names` list returned by the function contains the `group_column` values in the order they appear in `unique_ids`.

---

## 5. Rescaling Methods

The raw AEM potential φ can have arbitrary magnitude depending on segment lengths, strengths, and the spatial extent of the domain. For numerical stability of the kriging system, the potential must be rescaled to be comparable in magnitude to the variogram sill.

Two rescaling methods are supported, controlled by the `rescaling_method` parameter:

### 5.1 Adaptive Rescaling (Default)

```
rescr = sill / max(|φ_group|)
```

The scaling factor is computed so that the maximum absolute value of the scaled potential equals the variogram sill. This normalises each group's potential independently.

- **Advantage:** Automatically adapts to the actual magnitude of the potential field; robust across different unit systems and domain sizes.
- **Use case:** Default for all new analyses.

If `max(|φ_group|) ≤ 1e-10` (effectively zero potential), the scaling factor defaults to 1.0 to avoid division by zero.

### 5.2 Fixed Rescaling (KT3D-style)

```
rescr = sill / 0.0001
```

A constant scaling factor is applied regardless of the actual potential magnitude. This matches the original KT3D Fortran implementation.

- **Advantage:** Reproducible, independent of data.
- **Disadvantage:** Can produce very large drift values if the raw potential is small, potentially destabilising the kriging matrix.
- **Use case:** Legacy compatibility only.

---

## 6. The Critical Training/Prediction Contract

> **This is the most important operational constraint for AEM drift.**

The scaling factors `rescr` are computed from the **training data** (the observation well locations). During prediction on a grid, the potential values at grid points will generally differ from those at training points — so if scaling factors were recomputed at prediction time, the drift columns would be on a different scale than during training. This would invalidate the kriging coefficients β.

**The rule:** Scaling factors computed during training **must** be reused during prediction.

### How It Works

**Training phase** — [`compute_linesink_drift_matrix()`](../AEM_drift.py:53) is called without `input_scaling_factors`:

```python
drift_matrix, term_names, scaling_factors = compute_linesink_drift_matrix(
    x_model, y_model, linesinks_gdf, group_col,
    transform_params, sill,
    rescaling_method='adaptive'
    # input_scaling_factors not provided → factors are computed from training data
)
```

The returned `scaling_factors` dict maps each `group_column` value to its computed `rescr`.

**Prediction phase** — the saved `scaling_factors` are passed back in:

```python
drift_matrix_pred, _, _ = compute_linesink_drift_matrix(
    x_grid, y_grid, linesinks_gdf, group_col,
    transform_params, sill,
    input_scaling_factors=scaling_factors   # ← reuse training factors
)
```

When `input_scaling_factors` is provided and contains the key for a group, that factor is used directly without recomputation:

```python
if input_scaling_factors and linesink_id in input_scaling_factors:
    rescr = input_scaling_factors[linesink_id]
```

**Consequence of not reusing factors:** The drift columns at prediction points would be scaled differently from training, causing the kriging system to apply incorrect drift coefficients and producing systematically biased predictions.

### Return Value

[`compute_linesink_drift_matrix()`](../AEM_drift.py:53) returns a 3-tuple:

```python
(drift_matrix, term_names, scaling_factors_used)
```

| Element | Type | Description |
|---|---|---|
| `drift_matrix` | `np.ndarray` shape `(n_points, n_groups)` | Scaled AEM potential columns |
| `term_names` | `list[str]` | Group names in column order |
| `scaling_factors_used` | `dict[str, float]` | Maps group name → scaling factor used |

---

## 7. The `apply_anisotropy` Toggle

When anisotropy is enabled in the variogram, observation well coordinates are transformed to model space before kriging (see [Anisotropy Theory](anisotropy.md)). The question then arises: should the linesink geometry also be transformed to model space when computing the AEM potential?

The `apply_anisotropy` parameter controls this behaviour:

### `apply_anisotropy = True` (default)

The linesink segment endpoints are transformed to model space using the same `transform_params` as the observation wells. The AEM potential is then computed in model space, consistent with the kriging coordinate system.

```python
current_transform = transform_params   # use anisotropy transform
x1_m, y1_m = apply_transform(p1_raw[0], p1_raw[1], current_transform)
x2_m, y2_m = apply_transform(p2_raw[0], p2_raw[1], current_transform)
```

**Physical meaning:** The river geometry is "stretched" in the same way as the observation data. The AEM potential field is computed in the isotropic model space, so it is consistent with the kriging distances.

### `apply_anisotropy = False`

The linesink segment endpoints are used in their **raw (geographic) coordinates**. The AEM potential is computed in raw space, even though the kriging system operates in model space.

```python
current_transform = None   # bypass transform → apply_transform returns raw coords
```

**Physical meaning:** The river geometry is preserved in its true geographic shape. This is the KT3D-style approach and may be appropriate when the AEM potential is intended to capture the physical river geometry without distortion.

**Interaction with polynomial drift:** When `apply_anisotropy=False` for linesink drift, the polynomial drift terms are still computed in model space (they use the transformed x, y coordinates). Only the AEM columns use raw coordinates.

### Configuration

In `config.json`, this is controlled via the `drift_terms.linesink_river` entry:

```json
"drift_terms": {
    "linesink_river": {
        "use": true,
        "apply_anisotropy": true
    }
}
```

Setting `"apply_anisotropy": false` uses raw coordinates for AEM computation.

---

## 8. Summary of Key Contracts

| Contract | Description |
|---|---|
| **Scaling factor persistence** | `scaling_factors` from training must be passed as `input_scaling_factors` during prediction |
| **Term name ordering** | `term_names` returned at training must match the column order used in the kriging system |
| **Coordinate space** | Controlled by `apply_anisotropy`; when `True`, segment endpoints are in model space; when `False`, raw space |
| **Zero-length segments** | Segments with L < 1e-6 contribute zero potential and are silently skipped |
| **Endpoint singularities** | Handled by perturbing ZZ by ±1e-10 when evaluation point is at a segment endpoint |
| **Group superposition** | All segments in a group are summed before scaling; scaling is per-group, not per-segment |

---

## 9. Reference

Strack, O. D. L. (1989). *Groundwater Mechanics*. Prentice Hall, Englewood Cliffs, NJ.

The complex potential formula implemented in [`compute_linesink_potential()`](../AEM_drift.py:7) follows the line sink element derivation in Strack (1989), adapted for numerical implementation with singularity protection.
