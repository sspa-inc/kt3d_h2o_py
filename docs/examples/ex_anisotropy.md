# Task 4.4 — EXAMPLE: Anisotropy Handling

**Script:** [`docs/examples/ex_anisotropy.py`](ex_anisotropy.py)  
**Output:** [`docs/examples/output/ex_anisotropy.svg`](output/ex_anisotropy.svg)

---

## Overview

This example demonstrates how UK_SSPA v2 handles **geometric anisotropy** — directional dependence of spatial correlation — using the coordinate pre-transformation approach.

**Scenario:**
- 20 synthetic observation points with **true anisotropic spatial correlation** generated via Cholesky decomposition of an anisotropic spherical covariance matrix
- `angle_major=30` deg (azimuth, N30E direction), `ratio=0.3`
- Major correlation along the direction perpendicular to `angle_major`
- Minor correlation along the `angle_major` direction (N30E)

---

## Critical: How `angle_major` Relates to the Correlation Axes

> **The `angle_major` parameter specifies the direction that gets STRETCHED in model space, which corresponds to the direction of SHORTER correlation (the minor axis). The major correlation axis is PERPENDICULAR to `angle_major`.**

This is because [`apply_transform()`](../../transform.py:64) works as follows:
1. Convert azimuth to arithmetic internally (`alpha = 90 - azimuth`)
2. Rotate using `coords @ R` — this maps the `angle_major` direction to the **Y-axis**
3. Scale Y by `1/ratio` — this **stretches** Y-axis distances, making them larger in model space
4. Larger model-space distances → variogram reaches its range sooner → **shorter effective correlation**

**Numerical verification** (from the script output):
```
NE (1,1) -> model dist=4.714 (stretched by 1/ratio=3.3)
NW (-1,1) -> model dist=1.414 (unchanged)
Ratio of distances: 3.33 (= 1/ratio)
```

**For `angle_major=30` deg (azimuth, N30E), `ratio=0.3`, `range=120`:**

| Direction | Maps to in model space | Scaling | Effective range in raw space | Role |
|---|---|---|---|---|
| N30E (30° azimuth = `angle_major`) | Y-axis | Stretched by 1/ratio = 3.3 | 120 * 0.3 = **36** | **Minor axis** |
| Perpendicular (120° azimuth) | X-axis | Unchanged | **120** | **Major axis** |

---

## Angle Convention

> All angles in UK_SSPA v2 are **azimuth** — measured clockwise from the positive Y-axis (North). This matches the KT3D SETROT convention.

| `angle_major` (azimuth) | Direction stretched | Major correlation direction |
|---|---|---|
| 0° | North (+Y) | East/West (perpendicular) |
| 30° | N30E | Perpendicular to N30E |
| 45° | NE | NW/SE (perpendicular) |
| 90° | East (+X) | North/South (perpendicular) |

**Conversion to/from arithmetic (internal):**
```
arithmetic = 90 - azimuth   (mod 360)
azimuth    = 90 - arithmetic (mod 360)
```

---

## Synthetic Data Generation

The example generates data with **true anisotropic spatial correlation** using the **same transform** as the production code:

1. Place 20 random points in a 200x200 domain
2. Transform coordinates to model space using [`apply_transform()`](../../transform.py:64) with the same `angle_major` and `ratio`
3. Compute pairwise distances in model space (where the field is isotropic)
4. Build a spherical covariance matrix using `range_major` in model space
5. Generate correlated values via **Cholesky decomposition**: `z = L @ N(0,1)` where `L L^T = C`

This ensures the synthetic data has the exact anisotropic structure that the kriging model expects.

---

## Anisotropy Parameters

```python
ANGLE_MAJOR = 30.0   # degrees azimuth (CW from North) — N30E direction
RATIO       = 0.3    # minor_range / major_range
RANGE_MAJOR = 120.0  # range in model space = effective range along perpendicular direction
```

- **Major range** = 120 units along the direction perpendicular to N30E
- **Minor range** = 120 * 0.3 = 36 units along N30E (`angle_major` direction)

---

## Pre-Transformation Steps

Given raw coordinates **X** = (x, y):

**Step 1 — Translate to centroid:**
```
X_c = X - center
```

**Step 2 — Convert azimuth to arithmetic (internal):**
```
alpha = 90 - angle_major = 90 - 30 = 60°
```

**Step 3 — Rotate using `coords @ R`:**
```
X_r = (X_c) @ R

R = [[cos(α), -sin(α)],
     [sin(α),  cos(α)]]
```
After rotation, the `angle_major` direction maps to the **Y-axis** and the perpendicular direction maps to the **X-axis**.

**Step 4 — Scale Y by `1/ratio`:**
```
X' = S * X_r,    S = [1.0, 1/ratio]
```
The Y-axis (the `angle_major` direction) is stretched, making the field isotropic in model space.

**Full formula:** `X' = S * ((X - center) @ R)`

---

## Why Clone the Variogram with `anisotropy_enabled=False`?

After pre-transforming the coordinates, the field is **already isotropic** in model space. If PyKrige's internal anisotropy were also enabled, the transformation would be applied **twice**.

```python
# Clone with anisotropy disabled — passed to PyKrige
vario_iso_clone = vario_aniso.clone()
vario_iso_clone.anisotropy_enabled = False
vario_iso_clone.anisotropy_ratio = 1.0
vario_iso_clone.angle_major = 0.0

# Build model in model space
uk_model = build_uk_model(x_model, y_model, z,
                          drift_matrix=None, variogram=vario_iso_clone)
```

> **Contract:** [`build_uk_model()`](../../kriging.py:46) checks `anisotropy_enabled`. When `False`, it sets `anisotropy_scaling=1.0` and `anisotropy_angle=0.0` in PyKrige.

---

## Prediction Pipeline

```python
# 1. Compute transform parameters from training data
params = get_transform_params(x_raw, y_raw, angle_deg=30.0, ratio=0.3)

# 2. Transform training coordinates to model space
x_model, y_model = apply_transform(x_raw, y_raw, params)

# 3. Build model in model space (clone with anisotropy_enabled=False)
uk_model = build_uk_model(x_model, y_model, z,
                          drift_matrix=None, variogram=vario_iso_clone)

# 4. Transform prediction grid to model space (same params!)
flat_x_model, flat_y_model = apply_transform(flat_x, flat_y, params)

# 5. Predict in model space
z_pred, variance = uk_model.execute("points", flat_x_model, flat_y_model)
```

---

## Output Figure

![Anisotropy Example](output/ex_anisotropy.svg)

The 6-panel figure shows:

| Panel | Description |
|---|---|
| Top-left | Raw point cloud with correlation ellipse: major axis perpendicular to N30E (blue, range=120), minor axis along N30E (red dashed, range=36), angle arc showing `angle_major=30°` azimuth |
| Top-center | Explanation panel: transform mechanics, how `angle_major` maps to the minor axis |
| Top-right | Difference map (aniso minus iso) showing where anisotropy changes the prediction |
| Bottom-left | Prediction **with** anisotropy — correlation elongated perpendicular to N30E |
| Bottom-center | Prediction **without** anisotropy — treats all directions equally |
| Bottom-right | Variance comparison: aniso variance (color) vs iso variance (blue contours) — aniso low-variance zones elongated perpendicular to N30E |

**Key observations:**
- The **prediction with anisotropy** shows smoother interpolation along the major correlation axis (perpendicular to N30E), correctly reflecting the directional structure
- The **variance panel** shows elongated low-variance zones along the major correlation direction around observation points
- The **isotropic prediction** treats all directions equally, producing circular influence zones

---

## Verification

```
Roundtrip error: 4.26e-14  (< 1e-12 threshold)
NE (1,1) -> model dist=4.714 (stretched by 1/ratio=3.3)
NW (-1,1) -> model dist=1.414 (unchanged)
```

---

## Key Takeaways

1. **`angle_major` specifies the direction that gets stretched** (the minor correlation axis). The major correlation axis is perpendicular to it.

2. For `angle_major=30°` (azimuth, N30E): N30E gets stretched (minor, range=36), the perpendicular direction is unchanged (major, range=120).

3. **Pre-transformation** converts raw coordinates to model space where the field is isotropic, then isotropic kriging is applied.

4. **Clone the variogram** with `anisotropy_enabled=False` before passing to [`build_uk_model()`](../../kriging.py:46) to prevent double-application.

5. **Both training and prediction** must use the same `transform_params` computed from training data.

6. The `ratio = minor_range / major_range`, so `ratio=0.3` means the minor range is 30% of the major range.

---

## Related Documentation

- Theory: [`docs/theory/anisotropy.md`](../theory/anisotropy.md)
- API: [`docs/api/transform.md`](../api/transform.md)
- V&V: [`docs/validation/vv_transform_roundtrip.py`](../validation/vv_transform_roundtrip.py)
- V&V: [`docs/validation/vv_anisotropy_consistency.py`](../validation/vv_anisotropy_consistency.py)
