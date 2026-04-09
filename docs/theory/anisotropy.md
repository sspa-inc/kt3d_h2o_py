# Anisotropy Theory and Implementation

## 1. What Is Geometric Anisotropy?

Spatial correlation in groundwater systems is often **directionally dependent**. For example, a glacial aquifer may have a much longer correlation length along the direction of ice flow than perpendicular to it. This is called **geometric anisotropy**: the shape of the variogram (specifically its range) varies with direction, but the sill and nugget remain constant.

Formally, the variogram in direction θ has range `a(θ)`, and the anisotropy is described by:

- **Major axis direction** (`angle_major`): the direction of maximum spatial correlation (longest range)
- **Major range** (`range`): the variogram range along the major axis
- **Minor range**: `ratio × range`, the variogram range along the perpendicular (minor) axis
- **Anisotropy ratio** (`ratio`): `minor_range / major_range`, constrained to `(0, 1]`

A ratio of 1.0 means the field is **isotropic** (equal correlation in all directions). A ratio of 0.5 means the minor-axis range is half the major-axis range.

---

## 2. The Pre-Transformation Approach

UK_SSPA v2 handles geometric anisotropy via **coordinate pre-transformation** rather than using PyKrige's internal anisotropy parameters. The approach is:

1. **Transform** the input coordinates from raw space to an isotropic **model space** using an affine transformation
2. **Run isotropic kriging** in model space (PyKrige's internal anisotropy is disabled)
3. **Invert** the transformation when converting model-space results back to raw space

This approach has several advantages:
- It is mathematically equivalent to anisotropic kriging
- It allows the same transformation to be applied consistently to drift terms, control points, and prediction grids
- It avoids numerical issues that can arise from PyKrige's internal anisotropy handling when combined with specified drift

> **Critical contract:** PyKrige's internal anisotropy parameters (`anisotropy_scaling`, `anisotropy_angle`) are explicitly set to their identity values (scaling=1, angle=0) after pre-transformation. See [`kriging.py`](../kriging.py) for the variogram clone with `anisotropy_enabled=False`.

---

## 3. The Transformation Formula

The full transformation from raw coordinates **X** to model coordinates **X′** is:

```
X′ = S ⊙ ((X − center) @ R)
```

Where:
- `center` = centroid of the observation data `[mean(x), mean(y)]`
- `R` = 2D rotation matrix (built from the internally-converted arithmetic angle)
- `S` = element-wise scaling vector `[1, 1/ratio]`
- `⊙` = element-wise multiplication
- `@` = matrix multiplication (coordinates as row vectors times rotation matrix)

This is implemented in [`transform.py`](../transform.py) via [`apply_transform()`](../transform.py).

### Angle Input and Internal Conversion

The user specifies `angle_major` as an **azimuth** (clockwise from North), matching the KT3D SETROT convention. Internally, the code converts to an arithmetic angle before building the rotation matrix:

```
alpha = 90 - azimuth    (internal conversion)
```

### Rotation Matrix

For the internally-converted angle `α` (in radians, arithmetic convention):

```
R = | cos(α)  -sin(α) |
    | sin(α)   cos(α) |
```

The forward transform applies coordinates as row vectors: `coords @ R`, which rotates the coordinate system so the major axis aligns with the X-axis.

### Scaling Vector

```
S = [1.0,  1/ratio]
```

The X-axis (major axis after rotation) is left unchanged. The Y-axis (minor axis after rotation) is stretched by `1/ratio` to equalize the correlation lengths.

---

## 4. Step-by-Step Transformation

Given observation coordinates `(x, y)`, `angle_major` (degrees, azimuth CW from North), and `ratio`:

### Step 1 — Translate to Centroid

```python
x_center = mean(x)
y_center = mean(y)
x_c = x - x_center
y_c = y - y_center
```

This centers the data so that rotation is about the data centroid, not the coordinate origin.

### Step 2 — Convert Azimuth to Arithmetic (Internal)

```python
alpha = 90 - angle_major   # internal conversion
theta = radians(alpha)
```

This converts the user-facing azimuth angle to the arithmetic angle used by the rotation matrix.

### Step 3 — Rotate Using `coords @ R`

```python
R = [[cos(theta), -sin(theta)],
     [sin(theta),  cos(theta)]]

# Row-vector multiplication: coords @ R
x_r =  cos(theta) * x_c + sin(theta) * y_c
y_r = -sin(theta) * x_c + cos(theta) * y_c
```

After this step, the **major axis of correlation** is aligned with the **positive X-axis** in the rotated frame.

### Step 4 — Scale the Minor Axis

```python
x_prime = x_r * 1.0          # major axis unchanged
y_prime = y_r * (1/ratio)    # minor axis stretched
```

After scaling, the correlation structure is **isotropic** in model space: the effective range is `major_range` in all directions.

---

## 5. Angle Convention — Explicit Statement

> **`angle_major` uses the azimuth convention: Clockwise (CW) from the positive Y-axis (North). This matches the KT3D SETROT convention.**

| `angle_major` (azimuth) | Arithmetic (internal) | Major axis direction |
|---|---|---|
| 0° | 90° | North (positive Y-axis) |
| 30° | 60° | N30E |
| 45° | 45° | Northeast |
| 90° | 0° | East (positive X-axis) |
| 180° | −90° / 270° | South (negative Y-axis) |
| 270° | 180° | West (negative X-axis) |

**Examples:**
- `angle_major = 0°` → the major axis of spatial correlation points **North**; internally converted to 90° arithmetic
- `angle_major = 30°` → the major axis points **N30E**; internally converted to 60° arithmetic
- `angle_major = 90°` → the major axis points **East**; internally converted to 0° arithmetic

> **Internal conversion:** `alpha = 90 − azimuth`. The standard rotation matrix is then built using `alpha`. See [`docs/glossary.md`](../glossary.md) for the full conversion table.

---

## 6. Diagram: Before and After Transformation

The following illustrates a point cloud with `angle_major = 60°` (azimuth, N60E direction), `ratio = 0.5`:

```
RAW SPACE                          MODEL SPACE
(major axis at 60° azimuth = N60E) (major axis aligned with X-axis)

    Y                                  Y'
    |   . .                            |
    |  . . . .                         |  . .
    | . . . . . .                      | . . .
    |  . . . . .                       | . . .
    |   . . .                          |  . .
    +------------- X                   +------------- X'

  Elongated at N60E direction          Elongated along X-axis
  (major range >> minor range)       (isotropic after Y-stretch)
```

**Transformation steps applied:**
1. Translate: subtract centroid → data centered at origin
2. Convert: azimuth 60° → arithmetic 30° (internal)
3. Rotate using `coords @ R` → major axis now points along +X
4. Scale Y by `1/0.5 = 2` → minor axis stretched to match major axis length

After step 4, the point cloud appears **circular** (isotropic) in model space, and standard isotropic kriging applies.

---

## 7. Inverse Transform

To convert model-space coordinates back to raw space (e.g., for output or visualization), the inverse transformation is applied via [`invert_transform_coords()`](../transform.py):

```
X = (X′ / S) @ R^T + center
```

Step-by-step:

```python
# 1. Un-scale
x_us = x_prime / 1.0
y_us = y_prime / (1/ratio)   # = y_prime * ratio

# 2. Un-rotate (transpose of R = inverse for orthogonal matrix)
# Using coords @ R^T
x_ur =  cos(theta) * x_us - sin(theta) * y_us
y_ur =  sin(theta) * x_us + cos(theta) * y_us

# 3. Un-translate
x = x_ur + x_center
y = y_ur + y_center
```

The inverse transform is used in [`predict_on_grid()`](../kriging.py) to convert grid coordinates from raw space to model space before prediction, and is also available for converting model-space outputs back to raw space.

---

## 8. Why PyKrige's Internal Anisotropy Is Disabled

PyKrige supports anisotropy natively via `anisotropy_scaling` and `anisotropy_angle` parameters. When UK_SSPA v2 uses pre-transformation, PyKrige's internal anisotropy **must be disabled** to avoid double-application:

| Scenario | Result |
|---|---|
| Pre-transform applied + PyKrige anisotropy enabled | **Double anisotropy** — incorrect, over-corrected |
| Pre-transform applied + PyKrige anisotropy disabled | **Correct** — isotropic kriging in model space |
| No pre-transform + PyKrige anisotropy enabled | Equivalent (but drift terms would need separate handling) |

In practice, the [`variogram`](../variogram.py) object is **cloned** with `anisotropy_enabled=False` before being passed to PyKrige. This clone preserves all other variogram parameters (sill, range, nugget, model type) but sets `anisotropy_scaling=1` and `anisotropy_angle=0`.

---

## 9. The `apply_anisotropy` Toggle for Linesink Drift

The `drift_terms.linesink_river` configuration supports an `apply_anisotropy` flag:

```json
"linesink_river": {
    "use": true,
    "apply_anisotropy": true
}
```

| `apply_anisotropy` | Behavior |
|---|---|
| `true` (default) | Linesink segment geometry is transformed to model space before computing the AEM potential field. The drift term is computed in model space, consistent with the kriging system. |
| `false` | Linesink geometry remains in **raw space** for AEM computation. The resulting potential field is computed in raw coordinates, then used as a drift column in the (model-space) kriging system. |

**Physical interpretation of `apply_anisotropy: false`:**

Use this when the river's hydraulic influence is better represented in raw geographic space — for example, when the river geometry itself defines the anisotropy direction and you do not want the AEM potential to be distorted by the coordinate transformation. This is an advanced option; the default (`true`) is appropriate for most cases.

> **Coordinate space contract:** When `apply_anisotropy=false`, the AEM drift columns are computed using raw coordinates, but polynomial drift columns and the kriging covariance structure are still computed in model space. The drift matrix columns are concatenated regardless, so the kriging system remains internally consistent — only the geometric interpretation of the AEM term differs.

---

## 10. Implementation Reference

| Function | File | Purpose |
|---|---|---|
| [`get_transform_params()`](../transform.py) | `transform.py` | Computes `center`, `R`, `S` from data coordinates and variogram parameters; converts azimuth to arithmetic internally |
| [`apply_transform()`](../transform.py) | `transform.py` | Applies forward transform: raw → model space via `coords @ R` then scale |
| [`invert_transform_coords()`](../transform.py) | `transform.py` | Applies inverse transform: model → raw space |
| [`variogram.clone()`](../variogram.py) | `variogram.py` | Creates a copy with `anisotropy_enabled=False` for passing to PyKrige |
| [`predict_on_grid()`](../kriging.py) | `kriging.py` | Transforms grid coordinates to model space before prediction |

---

## See Also

- [`docs/glossary.md`](../glossary.md) — Angle convention, coordinate spaces, key terminology
- [`docs/api/transform.md`](../api/transform.md) — Full API reference for `transform.py`
- [`docs/theory/polynomial-drift.md`](polynomial-drift.md) — How drift terms are computed in model space
- [`docs/validation/vv_transform_roundtrip.py`](../validation/vv_transform_roundtrip.py) — V&V: roundtrip accuracy of forward + inverse transform
- [`docs/validation/vv_anisotropy_consistency.py`](../validation/vv_anisotropy_consistency.py) — V&V: equivalence of pre-transform approach vs PyKrige internal anisotropy
