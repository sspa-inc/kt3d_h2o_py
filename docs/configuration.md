# Configuration Reference

This document describes every key in `config.json`. The file is loaded and validated by [`load_config()`](../data.py:36) at startup. All four top-level sections (`data_sources`, `variogram`, `drift_terms`, `grid`) are **required**.

---

## Table of Contents

1. [data\_sources.observation\_wells](#1-data_sourcesobservation_wells)
2. [data\_sources.linesink\_river](#2-data_sourceslinesink_river)
3. [variogram](#3-variogram)
4. [drift\_terms](#4-drift_terms)
5. [grid](#5-grid)
6. [min\_separation\_distance](#6-min_separation_distance)
7. [output](#7-output)
8. [cross\_validation](#8-cross_validation)
9. [Minimal Config Example](#9-minimal-config-example)
10. [Full-Featured Config Example](#10-full-featured-config-example)
11. [Decision Table: drift\_terms.linesink\_river](#11-decision-table-drift_termslinesink_river)

---

## 1. `data_sources.observation_wells`

Primary point data source containing measured water levels.

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `data_sources.observation_wells.path` | string | **Yes** | — | Any valid file path | Path to the observation wells shapefile (`.shp`). Must be a Point geometry shapefile. | CRS must be consistent with all other input shapefiles. The tool does not reproject. |
| `data_sources.observation_wells.water_level_col` | string | **Yes** | — | Any column name present in the shapefile | Name of the attribute column containing measured water level values (numeric, no nulls). | Used throughout the pipeline as the primary response variable `z`. |

---

## 2. `data_sources.linesink_river`

Configuration for the linesink river shapefile used as an AEM drift source and/or for generating synthetic control points. This entire section is **optional**; omit it if no linesink drift is needed.

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `data_sources.linesink_river.path` | string | Conditional | — | Any valid file path | Path to the linesink river shapefile (`.shp`). Must be LineString or MultiLineString geometry. | Required when `drift_terms.linesink_river` is enabled or `control_points.enabled` is `true`. |
| `data_sources.linesink_river.group_column` | string | Conditional | `"DriftTerm"` | Any column name present in the shapefile | Attribute column that groups line segments into named linesink features. Segments sharing the same value are summed into one drift term. | Determines the column names in `term_names` for AEM drift terms. |
| `data_sources.linesink_river.strength_col` | string | No | `"resistance"` | Any numeric column name | Attribute column containing the hydraulic strength (resistance) of each segment. | Passed to [`compute_linesink_drift_matrix()`](../AEM_drift.py:53) as `strength_col`. |
| `data_sources.linesink_river.rescaling_method` | string | No | `"adaptive"` | `"adaptive"`, `"fixed"` | Controls how AEM potential values are normalized. `"adaptive"` scales each group so its maximum potential equals the variogram sill. `"fixed"` uses a KT3D-style constant: `sill / 0.0001`. | Scaling factors computed during training **must** be reused during prediction. See [Decision Table](#11-decision-table-drift_termslinesink_river). |

### 2a. `data_sources.linesink_river.control_points`

Sub-object controlling generation of synthetic control points along river line features. Control points are added to the observation dataset with interpolated water level elevations.

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `control_points.enabled` | boolean | No | `false` | `true`, `false` | Whether to generate synthetic control points along the linesink geometry. | When `true`, `z_start_col` and `z_end_col` must be present in the shapefile. |
| `control_points.spacing` | float | Conditional | — | `> 0` | Distance (in CRS units) between generated control points along each line segment. | Required when `enabled` is `true`. |
| `control_points.z_start_col` | string | Conditional | — | Any numeric column name | Attribute column containing the water level elevation at the **start** vertex of each segment. | Required when `enabled` is `true`. Used to linearly interpolate elevations along the segment. |
| `control_points.z_end_col` | string | Conditional | — | Any numeric column name | Attribute column containing the water level elevation at the **end** vertex of each segment. | Required when `enabled` is `true`. |
| `control_points.nugget_override` | float | No | `null` | `>= 0` and `< sill` | If set, overrides the variogram nugget for control points only. Useful when control points are considered more certain than observation wells. | Overrides `variogram.nugget` locally for control point rows only. |
| `control_points.avoid_vertices` | boolean | No | `true` | `true`, `false` | When `true`, generated points are placed away from the exact start/end vertices of each segment to avoid numerical issues. | Prevents duplicate or near-duplicate coordinates at segment junctions. |
| `control_points.perpendicular_offset` | float | No | `0.0` | Any float | Distance (in CRS units) to offset generated control points perpendicularly from the line. Useful to prevent control points from sitting exactly on the river centerline. | A value of `0.0` places points directly on the line. |

---

## 3. `variogram`

Defines the spatial correlation model. All three numeric parameters (`sill`, `range`, `nugget`) are **required** and validated by [`load_config()`](../data.py:36) and [`variogram._validate_basic_parameters()`](../variogram.py:38).

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `variogram.model` | string | No | `"spherical"` | `"spherical"`, `"exponential"`, `"gaussian"`, `"linear"` | The variogram model type. See `docs/theory/variogram-models.md` for equations. | Determines the shape of the spatial correlation function used in kriging. |
| `variogram.sill` | float | **Yes** | — | `> 0` | Total sill (variance at large lag distances). Must be strictly positive. | `nugget` must be strictly less than `sill`. The partial sill is `psill = sill - nugget`. |
| `variogram.range` | float | **Yes** | — | `> 0` | Correlation range in CRS units. For spherical and linear models, semivariance reaches the sill at exactly this distance. For exponential and gaussian, it is approached asymptotically. | Used in the rescaling factor computation: `resc = sqrt(sill / max(radsqd, range²))`. |
| `variogram.nugget` | float | **Yes** | — | `>= 0` and `< sill` | Nugget effect (discontinuity at the origin). Represents micro-scale variability or measurement error. | Must satisfy `0 <= nugget < sill`. |

### 3a. `variogram.anisotropy`

Controls geometric anisotropy via coordinate pre-transformation. When enabled, coordinates are rotated and scaled before kriging so the field appears isotropic in model space. See `docs/theory/anisotropy.md`.

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `variogram.anisotropy.enabled` | boolean | No | `false` | `true`, `false` | Whether to apply anisotropy pre-transformation. | When `true`, `ratio` and `angle_major` are validated by [`_validate_anisotropy()`](../variogram.py:48). PyKrige's internal anisotropy is disabled to avoid double-application. |
| `variogram.anisotropy.ratio` | float | Conditional | `1.0` | `(0, 1]` | Anisotropy ratio: `minor_range / major_range`. A value of `0.5` means the minor range is half the major range. Must be in `(0, 1]`. | Required when `enabled` is `true`. The Y-axis (minor axis in model space) is stretched by `1/ratio` during transformation. |
| `variogram.anisotropy.angle_major` | float | Conditional | `0.0` | `[0, 360)` | Direction of the **major axis** of spatial correlation, in **azimuth degrees** (clockwise from North). `0°` = North, `90°` = East, `45°` = Northeast. **This matches the KT3D SETROT convention.** | Required when `enabled` is `true`. Internally converted to arithmetic via `alpha = 90 - azimuth` before building the rotation matrix. To convert from arithmetic angle: `azimuth = 90 - arithmetic` (mod 360). |

### 3b. `variogram.advanced`

Optional tuning parameters for the kriging search neighborhood. All default to `null` (no limit).

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `variogram.advanced.search_radius` | float or null | No | `null` | `> 0` or `null` | Maximum search radius for neighboring points during kriging. Points beyond this distance are excluded. `null` means no limit. | Validated: if not null, must be `> 0`. |
| `variogram.advanced.max_neighbors` | integer or null | No | `null` | `>= 1` or `null` | Maximum number of neighboring points to use in the kriging system. `null` means use all points within `search_radius`. | Reduces computation time for large datasets. |
| `variogram.advanced.min_neighbors` | integer or null | No | `null` | `>= 1` or `null` | Minimum number of neighboring points required to produce a prediction. `null` means no minimum. | If fewer than `min_neighbors` points are found, the prediction at that location may be skipped or flagged. |
| `variogram.advanced.effective_range_convention` | boolean | No | `true` | `true`, `false` | When `true`, the `range` parameter is interpreted as the **effective range** (where semivariance reaches ~95% of sill for exponential/gaussian models). When `false`, it is the raw scale parameter. | Affects how `range` is passed to PyKrige internally. |

---

## 4. `drift_terms`

Controls which deterministic trend (drift) terms are included in the Universal Kriging model. All polynomial terms default to `false`. Omitting a key is equivalent to `false`.

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `drift_terms.linear_x` | boolean | No | `false` | `true`, `false` | Include a linear drift term in the X direction: `D = resc * x`. | Computed in model-space coordinates (after anisotropy transformation if enabled). |
| `drift_terms.linear_y` | boolean | No | `false` | `true`, `false` | Include a linear drift term in the Y direction: `D = resc * y`. | Computed in model-space coordinates. |
| `drift_terms.quadratic_x` | boolean | No | `false` | `true`, `false` | Include a quadratic drift term in the X direction: `D = resc * x²`. | Computed in model-space coordinates. |
| `drift_terms.quadratic_y` | boolean | No | `false` | `true`, `false` | Include a quadratic drift term in the Y direction: `D = resc * y²`. | Computed in model-space coordinates. |
| `drift_terms.linesink_river` | boolean **or** object | No | `false` | See [Decision Table](#11-decision-table-drift_termslinesink_river) | Controls whether AEM linesink drift is included. Can be a simple boolean or a dict with `use` and `apply_anisotropy` keys. | Requires `data_sources.linesink_river.path` to be set and valid. |

> **Term ordering contract:** Polynomial drift columns are always assembled in the fixed order `[linear_x, linear_y, quadratic_x, quadratic_y]`, regardless of the order keys appear in the config dict. AEM columns follow after polynomial columns. This order must be identical between training and prediction.

---

## 5. `grid`

Defines the prediction grid in **raw space** (original CRS coordinates). The grid is transformed to model space internally before prediction.

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `grid.x_min` | float | **Yes** | — | `< x_max` | Minimum X coordinate of the prediction grid (raw space, CRS units). | Validated: `x_min < x_max`. |
| `grid.x_max` | float | **Yes** | — | `> x_min` | Maximum X coordinate of the prediction grid (raw space, CRS units). | Validated: `x_max > x_min`. |
| `grid.y_min` | float | **Yes** | — | `< y_max` | Minimum Y coordinate of the prediction grid (raw space, CRS units). | Validated: `y_min < y_max`. |
| `grid.y_max` | float | **Yes** | — | `> y_min` | Maximum Y coordinate of the prediction grid (raw space, CRS units). | Validated: `y_max > y_min`. |
| `grid.resolution` | float | **Yes** | — | `> 0` | Grid cell size in CRS units. A smaller value produces a finer grid but increases computation time. | Validated: `resolution > 0`. The number of grid points is approximately `((x_max - x_min) / resolution) * ((y_max - y_min) / resolution)`. |

---

## 6. `min_separation_distance`

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `min_separation_distance` | float | No | — | `>= 0` | Minimum allowed distance (in CRS units) between any two data points. Points closer than this threshold are deduplicated by [`remove_duplicate_points()`](../data.py:134), retaining the first occurrence. | Applied after merging observation wells and control points. Set to `0` or omit to disable deduplication. |

---

## 7. `output`

Controls what outputs are generated after prediction. All keys are optional.

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `output.generate_map` | boolean | No | `true` | `true`, `false` | Whether to display an interactive matplotlib map of the predicted surface and variance. | When `false`, no map window is opened. |
| `output.export_contours` | boolean | No | `false` | `true`, `false` | Whether to export contour lines as a shapefile. | When `true`, `contour_interval` and `contour_output_path` must be set. |
| `output.contour_interval` | float | Conditional | `1.0` | `> 0` | Contour line interval in the same units as the water level values. | Used only when `export_contours` is `true`. |
| `output.contour_output_path` | string | Conditional | `"contours.shp"` | Any valid file path | Output path for the contour shapefile. Parent directory must exist. | Used only when `export_contours` is `true`. Output is a LineString shapefile. |
| `output.export_points` | boolean | No | `false` | `true`, `false` | Whether to export the merged observation + control point dataset as a shapefile. | When `true`, `points_output_path` must be set. |
| `output.points_output_path` | string | Conditional | `"observation_points.shp"` | Any valid file path | Output path for the auxiliary points shapefile. | Used only when `export_points` is `true`. Output is a Point shapefile with columns `x`, `y`, `h`. |

---

## 8. `cross_validation`

| Key Path | Type | Required | Default | Allowed Values | Description | Interactions |
|---|---|---|---|---|---|---|
| `cross_validation.enabled` | boolean | No | `false` | `true`, `false` | Whether to run Leave-One-Out Cross-Validation (LOOCV) after model training. Results (RMSE, MAE, Q1, Q2) are logged but not written to file. | Adds computation time proportional to the number of data points. Requires at least 3 data points. |

---

## 9. Minimal Config Example

Ordinary kriging with no drift, no anisotropy, no contour export:

```json
{
  "data_sources": {
    "observation_wells": {
      "path": "data/wells.shp",
      "water_level_col": "water_level"
    }
  },
  "variogram": {
    "model": "spherical",
    "sill": 1.0,
    "range": 1000.0,
    "nugget": 0.05
  },
  "drift_terms": {
    "linear_x": false,
    "linear_y": false,
    "quadratic_x": false,
    "quadratic_y": false,
    "linesink_river": false
  },
  "grid": {
    "x_min": 0.0,
    "x_max": 10000.0,
    "y_min": 0.0,
    "y_max": 10000.0,
    "resolution": 100.0
  }
}
```

This is the simplest valid configuration. The `output`, `min_separation_distance`, and `cross_validation` sections are omitted (all defaults apply).

---

## 10. Full-Featured Config Example

Anisotropic Universal Kriging with polynomial drift, AEM linesink drift, control points, and contour export:

```json
{
  "data_sources": {
    "observation_wells": {
      "path": "data/wells.shp",
      "water_level_col": "water_level"
    },
    "linesink_river": {
      "path": "data/rivers.shp",
      "group_column": "DriftTerm",
      "strength_col": "resistance",
      "rescaling_method": "adaptive",
      "control_points": {
        "enabled": true,
        "spacing": 300.0,
        "z_start_col": "UpElev",
        "z_end_col": "DNElev",
        "nugget_override": 0.5,
        "avoid_vertices": true,
        "perpendicular_offset": 50.0
      }
    }
  },
  "variogram": {
    "model": "spherical",
    "sill": 1.0,
    "range": 5000.0,
    "nugget": 0.0,
    "anisotropy": {
      "enabled": true,
      "ratio": 0.5,
      "angle_major": 45.0
    },
    "advanced": {
      "search_radius": null,
      "max_neighbors": null,
      "min_neighbors": null,
      "effective_range_convention": true
    }
  },
  "drift_terms": {
    "linear_x": true,
    "linear_y": true,
    "quadratic_x": false,
    "quadratic_y": false,
    "linesink_river": {
      "use": true,
      "apply_anisotropy": true
    }
  },
  "grid": {
    "x_min": -20000.0,
    "x_max": 5000.0,
    "y_min": -20000.0,
    "y_max": 5000.0,
    "resolution": 50.0
  },
  "min_separation_distance": 10.0,
  "output": {
    "generate_map": false,
    "export_contours": true,
    "contour_interval": 10.0,
    "contour_output_path": "output/contours.shp",
    "export_points": true,
    "points_output_path": "output/points.shp"
  },
  "cross_validation": {
    "enabled": true
  }
}
```

In this example:
- `angle_major: 45.0` means the major axis of spatial correlation points **Northeast** (45° azimuth, clockwise from North).
- `ratio: 0.5` means the minor range is half the major range.
- `linesink_river.apply_anisotropy: true` means the linesink geometry is transformed to model space before computing the AEM potential.
- Polynomial drift columns are assembled in fixed order: `[linear_x, linear_y]` (quadratic terms disabled).

---

## 11. Decision Table: `drift_terms.linesink_river`

The `linesink_river` drift term accepts three forms:

| Value | Type | `use_linesink` | `apply_anisotropy` | Behavior |
|---|---|---|---|---|
| `false` | boolean | `false` | N/A | AEM linesink drift is **disabled**. No AEM columns are added to the drift matrix. |
| `true` | boolean | `true` | `true` (default) | AEM linesink drift is **enabled**. Linesink geometry is transformed to model space (anisotropy applied) before computing the AEM potential field. |
| `{"use": true, "apply_anisotropy": true}` | object | `true` | `true` | Same as `true` above. Explicit form. |
| `{"use": true, "apply_anisotropy": false}` | object | `true` | `false` | AEM linesink drift is **enabled**, but linesink geometry remains in **raw space**. The AEM potential is computed using original (untransformed) coordinates. Polynomial drift and kriging still use model-space coordinates. Use this when the river geometry should not be distorted by the anisotropy transformation. |
| `{"use": false, "apply_anisotropy": true}` | object | `false` | N/A | AEM linesink drift is **disabled** (same as `false`). |

> **Critical contract:** When `use_linesink` is `true`, the scaling factors computed during training are stored as `trained_scaling_factors` and **must** be passed to [`predict_on_grid()`](../kriging.py:260) via the `scaling_factors` parameter. Failure to do so will cause the prediction-time AEM potential to be rescaled independently, producing inconsistent drift columns between training and prediction.

> **Source references:** [`main.py:380-436`](../main.py:380), [`data.py:36-128`](../data.py:36), [`variogram.py:6-52`](../variogram.py:6)
