# Glossary and Conventions

This document defines all domain terms, coordinate conventions, and angle definitions used in UK_SSPA v2. A reader should be able to determine the meaning of any term and the orientation of any angle without consulting source code.

---

## Key Terminology

| Term | Canonical Meaning |
|---|---|
| **Observation wells** | Primary point data with measured water levels; loaded from a Point shapefile. |
| **Control points** | Synthetic points generated along line features (rivers) with interpolated elevations; used to constrain the kriging surface near rivers. |
| **Linesink** | An Analytic Element Method (AEM) line segment that generates a potential field used as a drift term; represents the hydraulic influence of a river or drain. |
| **Drift term** | A deterministic trend function subtracted from the kriging residual before fitting; can be polynomial (linear/quadratic) or AEM-based. |
| **Rescaling factor (`resc`)** | A scaling constant applied to drift terms for numerical stability; computed as `resc = sqrt(sill / max(radsqd, range²))` where `radsqd` is the squared maximum radius of the data extent. A safety floor at `range²` prevents instability when the data extent is small relative to the correlation range. |
| **Scaling factors (AEM)** | Per-linesink-group multipliers that normalize the raw AEM potential to be comparable to the variogram sill. These are computed during model training and **must** be persisted and reused during prediction to ensure consistency. |
| **`term_names`** | Ordered list of drift column identifiers. Polynomial terms use fixed names (`linear_x`, `linear_y`, `quadratic_x`, `quadratic_y`); AEM terms use the `group_column` value from the shapefile. The order must be identical between training and prediction. |

---

## Coordinate System and Angle Convention

All coordinates are Cartesian X-Y in the units of the input shapefile CRS. The tool does **not** reproject data; all inputs must share the same CRS.

### Angle Definition

> **All angles in this tool are AZIMUTH angles: clockwise (CW) from the positive Y-axis (North).**
>
> This matches the **KT3D SETROT** convention used in GSLIB/Fortran geostatistics software.

| Property | Convention |
|---|---|
| Coordinate system | Cartesian X-Y; units match input shapefile CRS |
| Angle definition | **Azimuth**: Clockwise from North (positive Y-axis) |
| Angle = 0° | Points along the **positive Y-axis** (North) |
| Angle = 90° | Points along the **positive X-axis** (East) |
| Angle = 180° | Points along the **negative Y-axis** (South) |
| Angle = 270° | Points along the **negative X-axis** (West) |

### Angle Diagram

```
                  0° (North, +Y)
                       |
                       |
  270° (West) ---------+--------- 90° (East, +X)
                       |
                       |
                 180° (South, -Y)

  Angles increase clockwise (CW).
  Example: angle_major = 30° means the major axis points N30E (30° east of north).
```

### Conversion: Azimuth ↔ Arithmetic (Internal)

Internally, the code converts the user-facing azimuth angle to an arithmetic angle for the rotation matrix:

```
arithmetic = 90 - azimuth   (mod 360)
azimuth    = 90 - arithmetic (mod 360)
```

**Example:** An azimuth of 30° (N30E) corresponds to arithmetic angle 60° internally.

| Azimuth (user input) | Arithmetic (internal) | Direction |
|---|---|---|
| 0° | 90° | North (+Y) |
| 30° | 60° | N30E |
| 45° | 45° | NE |
| 90° | 0° | East (+X) |
| 180° | −90° / 270° | South (−Y) |
| 270° | 180° | West (−X) |

---

## Anisotropy Parameters

| Parameter | Definition |
|---|---|
| `angle_major` | The azimuth angle (CW from North) of the **major axis** of spatial correlation. `angle_major = 0°` means the major axis points North; `angle_major = 90°` means it points East. |
| `anisotropy_ratio` | `minor_range / major_range`, constrained to `(0, 1]`. A ratio of 1.0 means isotropic (equal range in all directions). A ratio of 0.5 means the minor range is **half** the major range. |
| Major axis alignment | After rotation by `angle_major`, the major axis aligns with the **X-axis** in model space. |
| Scaling direction | The **Y-axis** (minor axis in model space) is stretched by `1/ratio` to make the field isotropic. |
| Transform order | Translate to centroid → Rotate by `angle_major` (internally converted to arithmetic) → Scale Y by `1/ratio`. |

**Clarification on `anisotropy_ratio`:** `ratio = minor_range / major_range`. If `major_range = 200` and `minor_range = 100`, then `ratio = 0.5`. A smaller ratio means stronger anisotropy (greater directional contrast in correlation length).

---

## Coordinate Spaces

The tool operates in two coordinate spaces:

| Space | Description | When Used |
|---|---|---|
| **Raw space** | Original coordinates from input shapefiles, in the CRS units of the data. | Data loading, output generation, grid definition, control point generation. |
| **Model space** | Coordinates after anisotropy transformation: translate to centroid, rotate by `angle_major` (converted internally to arithmetic), scale Y by `1/ratio`. | Kriging training, polynomial drift computation, prediction (when anisotropy is enabled). |

**Important:** The prediction grid is defined in **raw space** (via `x_min`, `x_max`, `y_min`, `y_max` in the config) but is transformed to model space internally before kriging prediction. Output coordinates are always in raw space.

When `anisotropy.enabled = false`, raw space and model space are identical (no transformation is applied).

The `apply_anisotropy` toggle on the linesink drift term controls whether AEM potential is computed in raw space or model space. When `apply_anisotropy = false`, raw coordinates are used for AEM computation even though model coordinates are used for polynomial drift and kriging.

---

## Variogram Terminology

| Term | Definition |
|---|---|
| **Sill** | The total variance of the process; the value the variogram approaches at large lag distances. `sill = nugget + psill`. |
| **Nugget** | The discontinuity at the origin; represents measurement error or micro-scale variability. Must be less than `sill`. |
| **Partial sill (`psill`)** | The structured variance component: `psill = sill - nugget`. |
| **Range** | The lag distance at which the variogram reaches (or effectively reaches) the sill; beyond this distance, observations are uncorrelated. |
| **Effective range** | For models that approach the sill asymptotically (exponential, gaussian), the effective range is the distance at which the variogram reaches 95% of the sill. The `effective_range_convention` parameter controls whether `range` is interpreted as the model parameter or the effective range. |
