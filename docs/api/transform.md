# API Reference: `transform.py`

This module provides affine coordinate transformation utilities for geometric anisotropy handling in UK_SSPA v2. The transformation converts coordinates from **raw space** (original CRS) to **model space** (isotropic, centroid-centered), and back.

> **Angle Convention:** The `angle_deg` parameter (and `variogram.angle_major`) is an **azimuth — clockwise from North, 0° = North** (KT3D convention). Internally, the code converts this to an arithmetic angle before constructing the rotation matrix: `theta = 90 - angle_deg`. This conversion is transparent to the caller; always supply azimuth values.

---

## Transformation Overview

The full forward transformation is:

```
X' = (R · (X - center)) ⊙ S
```

Where:
- `X` = row vector of raw coordinates `[x, y]`
- `center` = centroid of the training data (computed from input arrays)
- `R` = 2×2 rotation matrix (rotates so the major axis aligns with the X-axis in model space)
- `S` = element-wise scaling vector `[1, 1/ratio]` (stretches the minor axis / Y-axis)
- `X'` = model-space coordinates

**Transform order:** Translate → Rotate → Scale

The inverse is:

```
X = (X' / S) · Rᵀ + center
```

---

## `get_transform_params()`

```python
get_transform_params(
    x: np.ndarray,
    y: np.ndarray,
    angle_deg: float,
    ratio: float,
) -> dict
```

Compute the transformation parameters from a set of input coordinates and anisotropy settings. The centroid is derived from the input arrays; the rotation matrix and scaling vector are derived from `angle_deg` and `ratio`.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `x` | `np.ndarray` | Yes | 1-D array of X coordinates (raw space). Used only to compute the centroid. |
| `y` | `np.ndarray` | Yes | 1-D array of Y coordinates (raw space). Used only to compute the centroid. |
| `angle_deg` | `float` | Yes | Major axis direction as an **azimuth: clockwise from North, 0° = North**. Internally converted to arithmetic angle via `theta = 90 - angle_deg`. |
| `ratio` | `float` | Yes | Anisotropy ratio `minor_range / major_range`. Must be in `(0, 1]`. If `ratio <= 0`, scaling defaults to `1.0` (no scaling). |

### Returns

`dict` with three keys:

| Key | Type | Shape | Description |
|---|---|---|---|
| `"center"` | `np.ndarray` | `(2,)` | Centroid `[mean(x), mean(y)]` of the input coordinates. Used as the translation origin. |
| `"R"` | `np.ndarray` | `(2, 2)` | Rotation matrix. Applied as `coords @ R` (row-vector convention). Rotates coordinates so the major axis aligns with the X-axis in model space. |
| `"S"` | `np.ndarray` | `(2,)` | Scaling vector `[1.0, 1/ratio]`. Applied element-wise after rotation. The X-axis (major axis in model space) is unscaled; the Y-axis (minor axis) is stretched by `1/ratio`. |

### Rotation Matrix Construction

The input azimuth `angle_deg` is converted to an arithmetic angle:

```
theta = radians(90 - angle_deg)
```

The standard 2-D CCW rotation matrix is then built:

```
R = [[cos(theta), -sin(theta)],
     [sin(theta),  cos(theta)]]
```

Applied as `coords_centered @ R` (row vectors), this rotates the data so the major axis (pointing in the azimuth direction) aligns with the X-axis in model space.

### Rotation Matrix Examples

| `angle_deg` (azimuth) | Direction | `theta` (arithmetic) | Effect in model space |
|---|---|---|---|
| `0°` | North | `90°` | Major axis was pointing North; after rotation it aligns with X-axis |
| `45°` | NE | `45°` | Major axis was pointing NE; after rotation it aligns with X-axis |
| `90°` | East | `0°` | Major axis was pointing East; `theta=0°`, R = identity |
| `135°` | SE | `-45°` | Major axis was pointing SE; after rotation it aligns with X-axis |

For `angle_deg = 90°` (major axis pointing East, `theta = 0°`):
```
R = [[1, 0],
     [0, 1]]   (identity — no rotation needed)
```

For `angle_deg = 0°` (major axis pointing North, `theta = 90°`):
```
R = [[cos90, -sin90],   =  [[0, -1],
     [sin90,  cos90]]       [1,  0]]
```

For `angle_deg = 45°` (major axis pointing NE, `theta = 45°`):
```
R ≈ [[ 0.707, -0.707],
     [ 0.707,  0.707]]
```

### Scaling Vector

After rotation, the major axis is aligned with X and the minor axis with Y. The scaling vector `S = [1.0, 1/ratio]` stretches the Y-axis (minor axis) by `1/ratio` to make the field isotropic:

- `ratio = 0.5` → `S = [1.0, 2.0]` — Y-axis stretched 2× (minor range was half the major range)
- `ratio = 1.0` → `S = [1.0, 1.0]` — no scaling (isotropic)

### Side Effects

Logs an `INFO` message with the computed center, angle, and ratio.

### Example

```python
import numpy as np
from transform import get_transform_params

x = np.array([100.0, 200.0, 150.0])
y = np.array([50.0, 80.0, 60.0])

# angle_deg=45 means major axis points NE (azimuth 45° CW from North)
params = get_transform_params(x, y, angle_deg=45.0, ratio=0.5)
print(params["center"])  # [150.0, 63.33...]
print(params["R"])       # 2x2 rotation matrix (theta = 45°)
print(params["S"])       # [1.0, 2.0]  (1/0.5 = 2.0)
```

---

## `apply_transform()`

```python
apply_transform(
    x: np.ndarray,
    y: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]
```

Apply the forward affine transformation to convert coordinates from **raw space** to **model space**.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `x` | `np.ndarray` | Yes | 1-D array of X coordinates in raw space. |
| `y` | `np.ndarray` | Yes | 1-D array of Y coordinates in raw space. |
| `params` | `dict` | Yes | Transform parameter dict as returned by [`get_transform_params()`](transform.py:14). If `None`, inputs are returned unchanged. |

### Returns

`tuple[np.ndarray, np.ndarray]` — `(x_prime, y_prime)` in **model space**.

### Implementation

```python
coords = np.column_stack((x, y))           # shape (N, 2)
coords_centered = coords - params["center"] # 1. Translate to centroid
coords_rotated  = np.dot(coords_centered, params["R"])  # 2. Rotate (row-vector × matrix)
coords_transformed = coords_rotated * params["S"]        # 3. Scale element-wise
```

The rotation is applied as `coords_centered @ R` (row vectors multiplied by the matrix on the right). This is equivalent to applying `Rᵀ` to column vectors.

### Coordinate Space

| Input | Output |
|---|---|
| Raw space (original CRS) | Model space (isotropic, centroid-centered) |

### Notes

- If `params` is `None`, returns `(x, y)` unchanged. This allows the transform to be safely called even when anisotropy is disabled.
- The same `params` dict computed from training data **must** be reused for all subsequent transforms (prediction grid, AEM linesink geometry). Do not recompute params from prediction-only points.

### Example

```python
import numpy as np
from transform import get_transform_params, apply_transform

x = np.array([0.0, 100.0, 200.0])
y = np.array([0.0, 0.0, 0.0])

# angle_deg=90 means major axis points East (azimuth 90°)
# theta = 90 - 90 = 0° → R = identity → no rotation
params = get_transform_params(x, y, angle_deg=90.0, ratio=0.5)
x_m, y_m = apply_transform(x, y, params)
# x_m: X coordinates in model space (major axis, unscaled)
# y_m: Y coordinates in model space (minor axis, stretched by 2×)
```

---

## `invert_transform_coords()`

```python
invert_transform_coords(
    x_prime: np.ndarray,
    y_prime: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]
```

Apply the inverse transformation to convert coordinates from **model space** back to **raw space**.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `x_prime` | `np.ndarray` | Yes | 1-D array of X coordinates in model space. |
| `y_prime` | `np.ndarray` | Yes | 1-D array of Y coordinates in model space. |
| `params` | `dict` | Yes | The **same** transform parameter dict used for the forward transform. If `None`, inputs are returned unchanged. |

### Returns

`tuple[np.ndarray, np.ndarray]` — `(x, y)` in **raw space** (original CRS).

### Implementation

```python
coords_prime     = np.column_stack((x_prime, y_prime))
coords_unscaled  = coords_prime / params["S"]              # 1. Un-scale
coords_unrotated = np.dot(coords_unscaled, params["R"].T)  # 2. Un-rotate (Rᵀ)
coords_original  = coords_unrotated + params["center"]     # 3. Un-translate
```

The inverse rotation uses `R.T` (transpose of the rotation matrix), which is the exact inverse because `R` is orthogonal (`R⁻¹ = Rᵀ`).

### Coordinate Space

| Input | Output |
|---|---|
| Model space (isotropic, centroid-centered) | Raw space (original CRS) |

### When to Use

Use `invert_transform_coords()` when you need to convert model-space results back to the original coordinate system. In the main pipeline, this is used to convert grid prediction coordinates from model space back to raw space for output generation.

### Notes

- If `params` is `None`, returns `(x_prime, y_prime)` unchanged.
- The inverse is exact — roundtrip error is at floating-point precision (`< 1e-12`).

### Example

```python
import numpy as np
from transform import get_transform_params, apply_transform, invert_transform_coords

x = np.array([100.0, 200.0, 300.0])
y = np.array([50.0, 100.0, 150.0])

params = get_transform_params(x, y, angle_deg=30.0, ratio=0.4)

# Forward transform
x_m, y_m = apply_transform(x, y, params)

# Inverse transform — recovers original coordinates
x_rec, y_rec = invert_transform_coords(x_m, y_m, params)

import numpy.testing as npt
npt.assert_allclose(x_rec, x, atol=1e-12)
npt.assert_allclose(y_rec, y, atol=1e-12)
```

---

## Orientation Diagram

The diagram below illustrates how `angle_deg` (azimuth) controls the major axis direction and how the transformation aligns it with the X-axis in model space.

```
RAW SPACE                              MODEL SPACE
(original coordinates)                 (after transform)

  Y (North)                              Y' (minor axis, stretched by 1/ratio)
  ^                                      ^
  |  * *                                 |  *  *  *  *
  | *   *   Major axis                   |
  |*     *  at azimuth=45° (NE)          |  *  *  *  *
  |       *                              |
  +-----------> X (East)                 +-----------> X' (major axis aligned)

angle_deg = 45° (azimuth, CW from North = NE direction)
ratio = 0.5  →  minor range is half the major range
               Y' axis is stretched by 1/0.5 = 2× to make field isotropic
```

### Azimuth → Model Space Alignment

| `angle_deg` (azimuth) | Major axis direction | After transform: major axis in model space |
|---|---|---|
| `0°` | North | X-axis |
| `45°` | NE | X-axis |
| `90°` | East | X-axis (no rotation needed) |
| `135°` | SE | X-axis |
| `180°` | South | X-axis |

In all cases, after the forward transform, the major axis of spatial correlation is aligned with the X-axis in model space, and the minor axis is aligned with (and stretched along) the Y-axis.

---

## Pipeline Integration

The transform functions are called in the following sequence in the main pipeline:

```python
# 1. Compute params from training data (using azimuth angle from variogram config)
params = get_transform_params(x_train, y_train,
                               angle_deg=vgm.angle_major,   # azimuth CW from North
                               ratio=vgm.anisotropy_ratio)

# 2. Transform training coordinates to model space
x_m, y_m = apply_transform(x_train, y_train, params)

# 3. Clone variogram and disable anisotropy (pre-transform already applied)
vgm_clone = vgm.clone()
vgm_clone.anisotropy_enabled = False

# 4. Build kriging model in model space
model = build_uk_model(x_m, y_m, z, drift_matrix, vgm_clone)

# 5. At prediction time: transform grid coordinates to model space using SAME params
x_grid_m, y_grid_m = apply_transform(x_grid, y_grid, params)

# 6. Predict in model space, then invert for output
x_out, y_out = invert_transform_coords(x_grid_m, y_grid_m, params)
```

**Critical contract:** The `params` dict computed from training data **must** be reused for all subsequent transforms. Computing new params from prediction-only points would produce a different centroid and incorrect results.
