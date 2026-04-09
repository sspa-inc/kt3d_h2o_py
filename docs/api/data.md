# API Reference: `data.py`

This module provides all data loading and preparation utilities for UK_SSPA v2. All functions return coordinates in **raw space** (original CRS from input shapefiles). No coordinate transformation is performed here.

---

## `load_config()`

```python
load_config(path: str) -> dict
```

Load and validate a JSON configuration file.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | `str` | Yes | Filesystem path to the JSON configuration file. |

### Returns

`dict` — Parsed and validated configuration dictionary. Structure mirrors the JSON file.

### Exceptions

| Exception | Condition |
|---|---|
| `FileNotFoundError` | File does not exist at `path`. |
| `ValueError` | File is not valid JSON, or numeric validation fails (e.g. `sill <= 0`). |
| `KeyError` | Required top-level keys are missing from the config. |

### Validation Rules

**Required top-level keys:** `data_sources`, `variogram`, `drift_terms`, `grid`

**Variogram constraints:**

| Parameter | Rule |
|---|---|
| `variogram.sill` | Must be a number and `> 0` |
| `variogram.range` | Must be a number and `> 0` |
| `variogram.nugget` | Must be a number, `>= 0`, and `< sill` |

**Grid constraints:**

| Parameter | Rule |
|---|---|
| `grid.x_min` | Must be `< grid.x_max` |
| `grid.y_min` | Must be `< grid.y_max` |
| `grid.resolution` | Must be `> 0` |

### Side Effects

Logs an `INFO` message on successful load. Logs `ERROR` messages before raising exceptions.

### Example

```python
from data import load_config

cfg = load_config("config.json")
print(cfg["variogram"]["sill"])  # e.g. 1.5
```

---

## `remove_duplicate_points()`

```python
remove_duplicate_points(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    min_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Remove co-located or near-duplicate points by clustering and averaging.

### Algorithm

1. Compute the full pairwise distance matrix between all points using `scipy.spatial.distance.cdist`.
2. Iterate through points in order. For each unprocessed point `i`, find all unprocessed points within `min_dist` (exclusive) — this forms a cluster.
3. Replace the cluster with its arithmetic mean in x, y, and h.
4. Mark all cluster members as processed.

This is a greedy, order-dependent algorithm. The first point in a cluster anchors the search radius.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `x` | `np.ndarray` | Yes | 1-D array of X coordinates. |
| `y` | `np.ndarray` | Yes | 1-D array of Y coordinates. |
| `h` | `np.ndarray` | Yes | 1-D array of head/value data. Must be same length as `x` and `y`. |
| `min_dist` | `float` | Yes | Distance threshold. Points closer than this are merged. If `<= 0`, inputs are returned unchanged (copied). |

### Returns

`tuple[np.ndarray, np.ndarray, np.ndarray]` — Three 1-D arrays `(x_clean, y_clean, h_clean)` with near-duplicate points merged.

### Exceptions

| Exception | Condition |
|---|---|
| `ValueError` | `min_dist` is `None` or cannot be converted to `float`. |

### Notes

- If `x` is empty, returns the empty arrays unchanged.
- If `min_dist <= 0`, returns copies of the input arrays with no merging.
- The merged position is the **mean** of all cluster members, not the position of the first point.

### Example

```python
import numpy as np
from data import remove_duplicate_points

x = np.array([0.0, 0.1, 10.0])
y = np.array([0.0, 0.0, 0.0])
h = np.array([5.0, 5.2, 8.0])

x_c, y_c, h_c = remove_duplicate_points(x, y, h, min_dist=1.0)
# x_c ≈ [0.05, 10.0], y_c ≈ [0.0, 0.0], h_c ≈ [5.1, 8.0]
```

---

## `load_observation_wells()`

```python
load_observation_wells(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Load observation well point data from a shapefile.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `config` | `dict` | Yes | Full validated configuration dictionary (as returned by [`load_config()`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/data.py)). Must contain `config["data_sources"]["observation_wells"]` with `path` and `water_level_col` keys. |

### Config Keys Used

| Key | Type | Required | Description |
|---|---|---|---|
| `data_sources.observation_wells.path` | `str` | Yes | Path to the point shapefile. |
| `data_sources.observation_wells.water_level_col` | `str` | Yes | Name of the column containing water level / head values. |

### Returns

`tuple[np.ndarray, np.ndarray, np.ndarray]` — Three 1-D arrays `(wx, wy, wh)`:

| Array | Description |
|---|---|
| `wx` | X coordinates in **raw space** (CRS of the shapefile). |
| `wy` | Y coordinates in **raw space**. |
| `wh` | Head/water-level values from `water_level_col`. |

### Coordinate Space

**Raw space.** Coordinates are taken directly from the shapefile geometry. No transformation is applied.

### Exceptions

| Exception | Condition |
|---|---|
| `KeyError` | `data_sources.observation_wells` missing from config, or `path`/`water_level_col` keys absent, or `water_level_col` not found in shapefile columns. |
| `FileNotFoundError` | Shapefile does not exist at the configured path. |
| `ValueError` | Length mismatch between extracted geometry coordinates and attribute values. |

### Notes

- If the shapefile contains no records, returns three empty arrays without raising an error.
- Non-Point geometries are handled by falling back to the geometry centroid.
- Null/empty geometries produce `NaN` coordinates in the output arrays.

---

## `load_line_features()`

```python
load_line_features(
    source_config: dict,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Read line features (e.g. rivers) from a shapefile and generate synthetic control points with linearly interpolated elevation values.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source_config` | `dict` | Yes | Source-specific configuration sub-dict (e.g. `config["data_sources"]["linesink_river"]`). |
| `config` | `dict` | Yes | Full configuration dictionary (used for global settings). |

### `source_config` Keys

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | `str` | Yes | — | Path to the line shapefile. |
| `control_points.enabled` | `bool` | No | `True` | If `False`, returns empty arrays immediately. |
| `control_points.spacing` | `float` | No | `50.0` | Distance between generated control points along each line segment. Must be `> 0`. |
| `control_points.z_start_col` | `str` | No | `None` | Column name for the elevation at the start of each line feature. |
| `control_points.z_end_col` | `str` | No | `None` | Column name for the elevation at the end of each line feature. |
| `control_points.nugget_override` | `float` | No | `0.0` | Nugget value assigned to all generated control points. |
| `control_points.avoid_vertices` | `bool` | No | `False` | If `True`, places points at segment midpoints avoiding vertices. If `False`, includes start and end vertices. |
| `control_points.perpendicular_offset` | `float` | No | `0.0` | Offset distance perpendicular to the line direction. Positive = left side. |

### Returns

`tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` — Four 1-D arrays `(cp_x, cp_y, cp_h, cp_n)`:

| Array | Description |
|---|---|
| `cp_x` | X coordinates of generated control points in **raw space**. |
| `cp_y` | Y coordinates of generated control points in **raw space**. |
| `cp_h` | Linearly interpolated head values between `z_start_col` and `z_end_col`. |
| `cp_n` | Nugget override values (constant `nugget_override` for all points). |

### Coordinate Space

**Raw space.** Coordinates are taken directly from the shapefile geometry. No transformation is applied.

### Exceptions

| Exception | Condition |
|---|---|
| `KeyError` | `path` key missing from `source_config`. |
| `FileNotFoundError` | Shapefile does not exist at the configured path. |
| `ValueError` | `control_points.spacing <= 0`. |

### Notes

- Line features where `z_start_col` or `z_end_col` values are `NaN` are **skipped entirely**. A warning is logged for each skipped feature.
- Shapefile column name truncation (10-character DBF limit) is handled automatically: if the full column name is not found, the first 10 characters are tried.
- `MultiLineString` geometries are decomposed into individual `LineString` parts.
- If no valid control points can be generated, returns four empty arrays and logs a warning.
- If `control_points.enabled` is `False`, returns four empty arrays immediately.

### Elevation Interpolation

For each control point at distance `d` along a line of total length `L`:

```
fraction = d / L
h = h_start + (h_end - h_start) * fraction
```

This assumes the line is digitized in the direction of flow (start → end).

---

## `prepare_data()`

```python
prepare_data(
    wx: np.ndarray,
    wy: np.ndarray,
    wh: np.ndarray,
    control_points_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Merge observation wells with one or more control point sources, then remove near-duplicate points.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `wx` | `np.ndarray` | Yes | X coordinates of observation wells. |
| `wy` | `np.ndarray` | Yes | Y coordinates of observation wells. |
| `wh` | `np.ndarray` | Yes | Head values of observation wells. |
| `control_points_list` | `list` | Yes | List of `(cx, cy, ch)` tuples, one per control point source. Pass `[]` if no control points. |
| `config` | `dict` | Yes | Full configuration dictionary. Uses `config["min_separation_distance"]` for duplicate removal. |

### Returns

`tuple[np.ndarray, np.ndarray, np.ndarray]` — Three 1-D arrays `(all_x, all_y, all_h)` after merging and duplicate removal.

### Merging Logic

1. Observation well arrays are placed first.
2. Each `(cx, cy, ch)` tuple from `control_points_list` is appended in order using `np.concatenate`.
3. The merged arrays are passed to [`remove_duplicate_points()`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/data.py) with `min_dist = config.get("min_separation_distance", 0.0)`.

### Exceptions

| Exception | Condition |
|---|---|
| `ValueError` | A `control_points_list` item is not a 3-tuple, or arrays within a tuple have mismatched lengths, or merged arrays have mismatched lengths. |

### Notes

- `None` inputs for `wx`, `wy`, `wh` are treated as empty arrays.
- `None` entries in `control_points_list` are silently skipped.
- If `min_separation_distance` is absent from config or `<= 0`, no duplicate removal is performed.
- The nugget override array `cp_n` returned by [`load_line_features()`](https://github.com/sspa-inc/kt3d_h2o_py/blob/main/data.py) is **not** passed to `prepare_data()` — it is handled separately in the main pipeline.

### Example

```python
from data import load_config, load_observation_wells, load_line_features, prepare_data

cfg = load_config("config.json")
wx, wy, wh = load_observation_wells(cfg)

cp_x, cp_y, cp_h, cp_n = load_line_features(cfg["data_sources"]["linesink_river"], cfg)

all_x, all_y, all_h = prepare_data(wx, wy, wh, [(cp_x, cp_y, cp_h)], cfg)
print(f"Total data points after merging: {len(all_x)}")
```
