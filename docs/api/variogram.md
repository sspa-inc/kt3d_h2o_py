# API Reference: `variogram.py`

The [`variogram`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/variogram.py) class encapsulates all variogram model parameters, validation, and semivariance computation for UK_SSPA v2.

---

## Class: `variogram`

```python
class variogram:
    def __init__(self, config=None, config_path="config.json")
```

### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `config` | `dict` | No | `None` | Pre-loaded configuration dictionary. If provided, `config_path` is ignored. |
| `config_path` | `str` | No | `"config.json"` | Path to JSON config file. Used only when `config` is `None`. |

If `config` is `None`, the constructor opens `config_path` and reads it with `json.load`. No validation of the file path is performed at this level — a `FileNotFoundError` or `json.JSONDecodeError` will propagate from the standard library.

---

## Attributes

All attributes are set during `__init__` and validated before the constructor returns.

### Core Variogram Parameters

| Attribute | Type | Source Key | Default | Description |
|---|---|---|---|---|
| `model_type` | `str` | `variogram.model` | `"spherical"` | Variogram model name. One of `"spherical"`, `"exponential"`, `"gaussian"`, `"linear"`. |
| `sill` | `float` | `variogram.sill` | `1.0` | Total sill (variance at large lag distances). Must be `> 0`. |
| `range_` | `float` | `variogram.range` | `1000.0` | Range parameter. For bounded models (spherical, linear), the semivariance reaches the sill at this distance. For unbounded models (exponential, gaussian), this is the practical range parameter. Must be `> 0`. |
| `nugget` | `float` | `variogram.nugget` | `0.0` | Nugget effect (discontinuity at the origin). Must be `>= 0` and `< sill`. |

### Anisotropy Parameters

| Attribute | Type | Source Key | Default | Description |
|---|---|---|---|---|
| `anisotropy_enabled` | `bool` | `variogram.anisotropy.enabled` | `False` | Whether geometric anisotropy is active. |
| `anisotropy_ratio` | `float` | `variogram.anisotropy.ratio` | `1.0` | `minor_range / major_range`. Must be in `(0, 1]` when anisotropy is enabled. A ratio of `1.0` means isotropic. |
| `angle_major` | `float` | `variogram.anisotropy.angle_major` | `0.0` | Direction of the major axis of spatial correlation. **Azimuth convention: Clockwise from North (KT3D convention). `0°` = North, `90°` = East.** Must be in `[0, 360)` when anisotropy is enabled. |

> **⚠ Angle Convention:** `angle_major` is stored and accepted as an **azimuth** (compass bearing — clockwise from North), following the KT3D convention. This is **not** arithmetic (CCW from East). Internally, the code converts to arithmetic via `theta = 90 - angle_major` before applying the rotation matrix. To convert: `arithmetic_CCW_from_East = 90 - azimuth_CW_from_North` (mod 360). See [`docs/glossary.md`](../glossary.md) for the full convention table.

> **Anisotropy Ratio:** `anisotropy_ratio = minor_range / major_range`. A ratio of `0.5` means the minor (short) range is half the major (long) range. The major axis is the direction of **greatest** spatial correlation (longest range). The minor axis is perpendicular to it.

### Advanced Parameters

| Attribute | Type | Source Key | Default | Description |
|---|---|---|---|---|
| `effective_range_convention` | `bool` | `variogram.advanced.effective_range_convention` | `True` | When `True`, the `range_` value is interpreted as the **effective range** (distance at which semivariance reaches ~95% of sill for exponential/gaussian). When `False`, it is the raw scale parameter. |
| `search_radius` | `float` or `None` | `variogram.advanced.search_radius` | `None` | Maximum search radius for neighbor selection during kriging. `None` means no limit. Must be `> 0` if set. |
| `max_neighbors` | `int` or `None` | `variogram.advanced.max_neighbors` | `None` | Maximum number of neighbors to use in kriging. `None` means no limit. |
| `min_neighbors` | `int` or `None` | `variogram.advanced.min_neighbors` | `None` | Minimum number of neighbors required for kriging at a point. `None` means no minimum. |

### Properties

| Property | Returns | Description |
|---|---|---|
| `model` | `str` | Alias for `model_type`. |
| `parameters` | `dict` | `{'sill': ..., 'range': ..., 'nugget': ...}` — convenience dict for passing to PyKrige. |

---

## Validation Methods

### `_validate_basic_parameters()`

Called automatically during `__init__`. Raises `ValueError` if:

| Condition | Error Message |
|---|---|
| `sill <= 0` | `"Variogram sill must be positive, got {sill}"` |
| `range_ <= 0` | `"Variogram range must be positive, got {range_}"` |
| `nugget < 0` | `"Variogram nugget must be non-negative, got {nugget}"` |
| `nugget >= sill` | `"Variogram nugget ({nugget}) must be less than total sill ({sill})"` |

### `_validate_anisotropy()`

Called automatically during `__init__` only when `anisotropy_enabled=True`. Raises `ValueError` if:

| Condition | Error Message |
|---|---|
| `anisotropy_ratio` not in `(0, 1]` | `"Anisotropy ratio must be in (0, 1], got {ratio}"` |
| `angle_major` not in `[0, 360)` | `"Angle major must be in [0, 360), got {angle_major}"` |

---

## Methods

### `calculate_variogram(h)`

```python
calculate_variogram(h: float) -> float
```

Compute the semivariance for a scalar isotropic lag distance `h`.

| Parameter | Type | Description |
|---|---|---|
| `h` | `float` | Isotropic lag distance. Must be `>= 0`. |

**Returns:** `float` — semivariance `γ(h)`.

**Note:** When `anisotropy_enabled=True`, this method still computes the isotropic semivariance. The caller is responsible for passing a pre-transformed (model-space) distance. For directional computation from raw-space lag vectors, use [`calculate_variogram_at_vector()`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/variogram.py) instead.

**Model equations** (where `psill = sill - nugget`):

| Model | Equation |
|---|---|
| `linear` | `γ(h) = nugget + (psill / range) * h` for `h ≤ range`, else `sill` |
| `exponential` | `γ(h) = nugget + psill * (1 - exp(-h / (range/3)))` |
| `spherical` | `γ(h) = nugget + psill * (1.5*(h/range) - 0.5*(h/range)³)` for `h ≤ range`, else `sill` |
| `gaussian` | `γ(h) = nugget + psill * (1 - exp(-(h / (range/√3))²))` |

---

### `calculate_variogram_at_vector(hx, hy)`

```python
calculate_variogram_at_vector(hx: float, hy: float) -> float
```

Compute the semivariance for a 2-D lag vector `(hx, hy)`, accounting for geometric anisotropy.

| Parameter | Type | Description |
|---|---|---|
| `hx` | `float` | X-component of the lag vector (in raw/original coordinate space). |
| `hy` | `float` | Y-component of the lag vector (in raw/original coordinate space). |

**Returns:** `float` — semivariance `γ(hx, hy)`.

**When `anisotropy_enabled=False`:** Computes `h = sqrt(hx² + hy²)` and delegates to `calculate_variogram(h)`.

**When `anisotropy_enabled=True`:** Applies the anisotropy transformation inline:

1. Convert `angle_major` from azimuth (CW from North) to arithmetic (CCW from East): `theta = radians(90 - angle_major)`
2. Rotate the lag vector to align the major axis with the X-axis:
   ```
   hx' =  hx * cos(theta) + hy * sin(theta)
   hy' = -hx * sin(theta) + hy * cos(theta)
   ```
3. Scale the minor axis (Y') to match the major range: `hy_scaled = hy' / ratio`
4. Compute effective isotropic distance: `h_eff = sqrt(hx'² + hy_scaled²)`
5. Return `calculate_variogram(h_eff)`

> **Note:** The rotation convention here matches [`transform.py`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/transform.py) — `theta = 90 - angle_major` converts the azimuth input to the arithmetic rotation angle applied to the coordinate system.

---

### `clone()`

```python
clone() -> variogram
```

Return a deep copy of this variogram object.

**Returns:** A new `variogram` instance with identical attribute values. Modifying the clone does not affect the original.

**Use case:** The main pipeline clones the variogram and sets `anisotropy_enabled=False` on the clone before passing it to PyKrige, to prevent double-application of anisotropy after the pre-transformation step has already been applied to the coordinates.

```python
vgm_clone = vgm.clone()
vgm_clone.anisotropy_enabled = False
# Pass vgm_clone to build_uk_model() when using pre-transformation
```

---

## Usage Examples

### Instantiate from config file

```python
from variogram import variogram

vgm = variogram(config_path="config.json")
print(vgm.model_type)          # e.g. "spherical"
print(vgm.sill)                # e.g. 1.5
print(vgm.range_)              # e.g. 500.0
print(vgm.nugget)              # e.g. 0.1
print(vgm.anisotropy_enabled)  # e.g. True
print(vgm.angle_major)         # e.g. 45.0  (azimuth: 45° CW from North = NE direction)
```

### Instantiate from dict

```python
from variogram import variogram

cfg = {
    "variogram": {
        "model": "exponential",
        "sill": 2.0,
        "range": 300.0,
        "nugget": 0.2,
        "anisotropy": {
            "enabled": True,
            "ratio": 0.4,
            "angle_major": 45.0   # azimuth: 45° CW from North = NE direction
                                   # internally converted to arithmetic: 90 - 45 = 45° CCW from East
        }
    }
}
vgm = variogram(config=cfg)
print(vgm.calculate_variogram(150.0))              # isotropic semivariance at h=150
print(vgm.calculate_variogram_at_vector(100, 50))  # directional semivariance
```

### Clone and disable anisotropy

```python
vgm_for_pykrige = vgm.clone()
vgm_for_pykrige.anisotropy_enabled = False
# vgm_for_pykrige.parameters returns {'sill': ..., 'range': ..., 'nugget': ...}
```
