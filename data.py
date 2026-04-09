"""
Data loading utilities for UK_SSPA v2 - Micro-Task 1.1

Provides:
 - load_config(path: str) -> dict

This module focuses on configuration loading and validation. Subsequent micro-tasks
will add the spatial data ingestion and duplicate-removal functions.
"""
from __future__ import annotations

import json
import os
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _ensure_number(value: Any, name: str) -> float:
    """Convert value to float and raise ValueError with a descriptive message on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Configuration parameter '{name}' must be a number. Got: {value!r}")


def load_config(path: str) -> Dict[str, Any]:
    """
    Load and validate a JSON configuration file.

    Validation rules (per micro-task 1.1):
    - File must exist and be valid JSON
    - Top-level keys required: ["data_sources", "variogram", "drift_terms", "grid"]
    - Variogram params: sill > 0, range > 0, nugget >= 0, nugget < sill
    - Grid params: x_min < x_max, y_min < y_max, resolution > 0

    Parameters
    ----------
    path : str
        Path to JSON config file.

    Returns
    -------
    dict
        Parsed and validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON or numeric validations fail.
    KeyError
        If required top-level keys are missing.
    """
    if not os.path.exists(path):
        logger.error("Config file not found: %s", path)
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in config file %s: %s", path, exc)
        raise ValueError(f"Invalid JSON in config file: {path}") from exc

    # Required top-level keys
    required_top_keys = ["data_sources", "variogram", "drift_terms", "grid"]
    missing = [k for k in required_top_keys if k not in cfg]
    if missing:
        logger.error("Missing required top-level config keys: %s", missing)
        raise KeyError(f"Missing required top-level config keys: {missing}")

    # Validate variogram
    variogram = cfg.get("variogram", {})
    if not isinstance(variogram, dict):
        raise ValueError("'variogram' must be an object/dictionary in config")

    try:
        sill = _ensure_number(variogram.get("sill"), "variogram.sill")
        rng = _ensure_number(variogram.get("range"), "variogram.range")
        nugget = _ensure_number(variogram.get("nugget"), "variogram.nugget")
    except ValueError as exc:
        logger.error("Variogram numeric validation failed: %s", exc)
        raise

    if sill <= 0.0:
        raise ValueError("Variogram parameter 'sill' must be > 0")
    if rng <= 0.0:
        raise ValueError("Variogram parameter 'range' must be > 0")
    if nugget < 0.0:
        raise ValueError("Variogram parameter 'nugget' must be >= 0")
    if nugget >= sill:
        raise ValueError("Variogram parameter 'nugget' must be less than 'sill'")

    # Validate grid
    grid = cfg.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("'grid' must be an object/dictionary in config")

    try:
        x_min = _ensure_number(grid.get("x_min"), "grid.x_min")
        x_max = _ensure_number(grid.get("x_max"), "grid.x_max")
        y_min = _ensure_number(grid.get("y_min"), "grid.y_min")
        y_max = _ensure_number(grid.get("y_max"), "grid.y_max")
        resolution = _ensure_number(grid.get("resolution"), "grid.resolution")
    except ValueError as exc:
        logger.error("Grid numeric validation failed: %s", exc)
        raise

    if x_min >= x_max:
        raise ValueError("Grid parameter 'x_min' must be less than 'x_max'")
    if y_min >= y_max:
        raise ValueError("Grid parameter 'y_min' must be less than 'y_max'")
    if resolution <= 0.0:
        raise ValueError("Grid parameter 'resolution' must be > 0")

    logger.info("Configuration loaded and validated from %s", path)
    return cfg


__all__ = ["load_config"]


def remove_duplicate_points(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    min_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove duplicate (co-located) points by clustering points within `min_dist`
    and replacing each cluster with its arithmetic mean (for x, y and h).

    Parameters
    ----------
    x, y, h : np.ndarray
        1-D arrays of coordinates and head values. Arrays must be of equal length.
    min_dist : float
        Distance threshold under which points are considered duplicates and
        will be merged. If `min_dist` <= 0 the inputs are returned unchanged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of cleaned x, y, h values where close points have been merged by mean.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    h = np.asarray(h)

    if x.size == 0:
        # empty input, return as-is
        return x, y, h

    if min_dist is None:
        raise ValueError("min_dist must be provided and be a number")

    try:
        min_dist = float(min_dist)
    except (TypeError, ValueError):
        raise ValueError("min_dist must be a numeric value")

    # If non-positive threshold, do not merge anything
    if min_dist <= 0.0:
        return x.copy(), y.copy(), h.copy()

    points = np.column_stack((x, y))
    dists = cdist(points, points)

    n = len(x)
    keep_mask = np.ones(n, dtype=bool)

    new_x: list[float] = []
    new_y: list[float] = []
    new_h: list[float] = []

    for i in range(n):
        if not keep_mask[i]:
            continue

        # cluster includes all currently unprocessed points within min_dist of point i
        cluster_indices = np.where((dists[i] < min_dist) & keep_mask)[0]

        # Defensive: ensure at least the point itself is included
        if cluster_indices.size == 0:
            cluster_indices = np.array([i], dtype=int)

        avg_x = float(np.mean(x[cluster_indices]))
        avg_y = float(np.mean(y[cluster_indices]))
        avg_h = float(np.mean(h[cluster_indices]))

        new_x.append(avg_x)
        new_y.append(avg_y)
        new_h.append(avg_h)

        # mark clustered points as processed
        keep_mask[cluster_indices] = False

    logger.info("Removed duplicates: reduced %d -> %d points using min_dist=%s", n, len(new_x), min_dist)

    return np.array(new_x), np.array(new_y), np.array(new_h)


__all__ = ["load_config", "remove_duplicate_points", "load_observation_wells", "load_line_features", "prepare_data"]


def load_observation_wells(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load observation well points from a point shapefile configured in `config`.

    Expects config["data_sources"]["observation_wells"] to be a dict with keys:
    - path: path to the shapefile
    - water_level_col: column name containing water-level / head values

    Returns
    -------
    (wx, wy, wh)
        Three 1-D numpy arrays with x, y coordinates and head values.

    Raises
    ------
    FileNotFoundError
        If the shapefile path does not exist.
    KeyError
        If required config keys are missing or the specified water level column
        is not present in the shapefile.
    ValueError
        If lengths of extracted arrays mismatch.
    """
    # Validate config structure and source entry
    try:
        source_conf = config["data_sources"]["observation_wells"]
    except Exception as exc:
        logger.error("Missing 'data_sources.observation_wells' in config: %s", exc)
        raise KeyError("Missing 'data_sources.observation_wells' entry in config") from exc

    path = source_conf.get("path")
    if not path:
        raise KeyError("'path' must be provided in data_sources.observation_wells")

    if not os.path.exists(path):
        logger.error("Observation wells file not found: %s", path)
        raise FileNotFoundError(f"Observation wells file not found: {path}")

    water_col = source_conf.get("water_level_col")
    if not water_col:
        raise KeyError("'water_level_col' must be provided in data_sources.observation_wells")

    logger.info("Loading observation wells from %s", path)

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        logger.error("Failed to read observation wells shapefile %s: %s", path, exc)
        raise

    # If empty file, return empty arrays
    if gdf is None or len(gdf) == 0:
        logger.info("Observation wells shapefile %s contains no records", path)
        return np.array([]), np.array([]), np.array([])

    if water_col not in gdf.columns:
        raise KeyError(f"water_level_col '{water_col}' not found in shapefile columns")

    # Extract coordinates robustly (handle non-Point geometries by using centroid)
    xs: list[float] = []
    ys: list[float] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            xs.append(np.nan)
            ys.append(np.nan)
            continue
        # For Point-like geometries use .x/.y, otherwise use centroid
        if hasattr(geom, "x") and hasattr(geom, "y"):
            try:
                xs.append(float(geom.x))
                ys.append(float(geom.y))
            except Exception:
                # Fallback to centroid
                c = geom.centroid
                xs.append(float(c.x))
                ys.append(float(c.y))
        else:
            c = geom.centroid
            xs.append(float(c.x))
            ys.append(float(c.y))

    wx = np.array(xs)
    wy = np.array(ys)

    wh = gdf[water_col].to_numpy()

    if not (len(wx) == len(wy) == len(wh)):
        logger.error("Mismatched lengths: coords=%d,%d values=%d", len(wx), len(wy), len(wh))
        raise ValueError("Mismatch between number of geometries and water-level values in observation wells shapefile")

    logger.info("Loaded %d observation wells from %s", len(wx), path)
    return wx, wy, wh


def load_line_features(source_config: dict, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read line features and generate control points with interpolated Z-values.
    Updated with debugging stats and safer NaN handling.
    """
    path = source_config.get("path")
    if not path:
        raise KeyError("'path' must be provided in line feature source configuration")

    if not os.path.exists(path):
        logger.error("Line features file not found: %s", path)
        raise FileNotFoundError(f"Line features file not found: {path}")

    control_points_config = source_config.get("control_points", {})
    if not control_points_config.get("enabled", True):
        logger.info("Control points disabled for this source")
        return np.array([]), np.array([]), np.array([]), np.array([])

    spacing = float(control_points_config.get("spacing", 50.0))
    nugget_val = float(control_points_config.get("nugget_override", 0.0))
    avoid_vertices = bool(control_points_config.get("avoid_vertices", False))
    offset_dist = float(control_points_config.get("perpendicular_offset", 0.0))
    
    if spacing <= 0.0:
        raise ValueError("control_point_spacing must be > 0")

    logger.info("Loading line features from %s with spacing=%s, nugget=%s", path, spacing, nugget_val)

    try:
        gdf = gpd.read_file(path)

        # if len(gdf) > 0:
        #     # 1. Print the source path to ensure it matches QGIS
        #     logger.info("DEBUG: Reading file from: %s", os.path.abspath(path))
            
        #     # 2. Print the WHOLE content of the first row
        #     first_row = gdf.iloc[0]
        #     logger.info("DEBUG: Full Row 0 Data:\n%s", first_row)
            
        #     # 3. Check specifically for the 'elevation' column we saw in the screenshot
        #     if 'elevation' in gdf.columns:
        #          logger.info("DEBUG: 'elevation' column value: %s", first_row['elevation'])


    except Exception as exc:
        logger.error("Failed to read line features shapefile %s: %s", path, exc)
        raise

    if gdf is None or len(gdf) == 0:
        logger.info("Line features shapefile %s contains no records", path)
        return np.array([]), np.array([]), np.array([]), np.array([])

    z_start_col = control_points_config.get("z_start_col")
    z_end_col = control_points_config.get("z_end_col")

    # Debug: Check if columns exist
    missing_cols = []
    if z_start_col and z_start_col not in gdf.columns:
        # Check for shapefile truncation (10 char limit)
        if z_start_col[:10] not in gdf.columns:
            missing_cols.append(z_start_col)
    if z_end_col and z_end_col not in gdf.columns:
         if z_end_col[:10] not in gdf.columns:
            missing_cols.append(z_end_col)
            
    if missing_cols:
        logger.warning("WARNING: The following elevation columns are missing from %s: %s. Control points may be skipped.", path, missing_cols)

    cp_x: list[float] = []
    cp_y: list[float] = []
    cp_h: list[float] = []
    cp_n: list[float] = []

    skipped_count = 0

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        parts = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []

        # Read start/end Z-values for this feature
        h_start = None
        h_end = None
        
        # Helper to safely get value or truncated value
        def get_val(col_name):
            if not col_name: return None
            if col_name in row.index: return row.get(col_name)
            if col_name[:10] in row.index: return row.get(col_name[:10])
            return None

        h_start = get_val(z_start_col)
        h_end = get_val(z_end_col)

        # Skip feature if data is missing (Prevent 0.0 insertion)
        if pd.isna(h_start) or pd.isna(h_end):
            skipped_count += 1
            continue

        h_start = float(h_start)
        h_end = float(h_end)

        for part in parts:
            if part is None: continue
            length = float(part.length)
            if length == 0.0: continue

            if avoid_vertices:
                num_points = int(np.floor(length / spacing))
                if num_points == 0:
                    distances = np.array([length / 2.0])
                else:
                    distances = np.linspace(spacing / 2.0, length - spacing / 2.0, num_points)
            else:
                # Include vertices: spacing is used but we ensure start (0) and end (length) are included
                num_points = int(np.ceil(length / spacing)) + 1
                distances = np.linspace(0, length, num_points)

            for d in distances:
                try:
                    point = part.interpolate(float(d))
                    px, py = float(point.x), float(point.y)

                    if offset_dist != 0.0:
                        # Get a point slightly further along to find the tangent/normal
                        # If at the very end, look slightly backward
                        d_eps = d + 0.1 if d + 0.1 <= length else d - 0.1
                        p_eps = part.interpolate(float(d_eps))
                        dx = float(p_eps.x) - px
                        dy = float(p_eps.y) - py
                        
                        # Normalize tangent
                        mag = np.sqrt(dx*dx + dy*dy)
                        if mag > 0:
                            dx /= mag
                            dy /= mag
                            
                            # Perpendicular vector (rotate 90 degrees)
                            # (nx, ny) = (-dy, dx)
                            nx, ny = -dy, dx
                            
                            # Apply offset
                            px += nx * offset_dist
                            py += ny * offset_dist

                    cp_x.append(px)
                    cp_y.append(py)

                    # Interpolate 'h' (head) value linearly
                    # Note: This assumes line digitization matches flow direction (Start -> End)
                    fraction = float(d) / length
                    stage = h_start + (h_end - h_start) * fraction
                    
                    cp_h.append(stage)
                    cp_n.append(nugget_val)
                except Exception:
                    continue

    # --- DEBUGGING STATS ---
    if len(cp_h) > 0:
        h_min = min(cp_h)
        h_max = max(cp_h)
        h_mean = sum(cp_h) / len(cp_h)
        logger.info("Control Points Stats: Count=%d, Min=%.2f, Max=%.2f, Mean=%.2f", len(cp_h), h_min, h_max, h_mean)
        if h_min == 0.0 and h_max > 100:
             logger.warning("WARNING: Some control points have 0.0 elevation while others are >100. Check data inputs.")
    else:
        logger.warning("No valid control points could be generated. Check column names and data.")
        
    if skipped_count > 0:
        logger.warning("Skipped %d line features due to missing Z-values (NaN).", skipped_count)

    return np.array(cp_x), np.array(cp_y), np.array(cp_h), np.array(cp_n)

def prepare_data(
    wx: np.ndarray,
    wy: np.ndarray,
    wh: np.ndarray,
    control_points_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | list,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge observation wells and multiple control-point sources, then remove
    duplicate/co-located points using `remove_duplicate_points()`.

    Parameters
    ----------
    wx, wy, wh : np.ndarray
        Arrays of observation well coordinates and head values.
    control_points_list : list
        A list of tuples, each tuple containing (cx, cy, ch) arrays for one source.
    config : dict
        Global configuration. May contain 'min_separation_distance' (float).

    Returns
    -------
    (all_x, all_y, all_h)
        Cleaned numpy arrays after merging and duplicate removal.
    """
    # Normalize inputs to numpy arrays
    wx = np.asarray(wx) if wx is not None else np.array([])
    wy = np.asarray(wy) if wy is not None else np.array([])
    wh = np.asarray(wh) if wh is not None else np.array([])

    arrays_x = [wx]
    arrays_y = [wy]
    arrays_h = [wh]

    # Append control point sources
    for src in control_points_list or []:
        if src is None:
            continue
        if not (isinstance(src, (list, tuple)) and len(src) == 3):
            # allow a single flat tuple of scalars -> skip
            raise ValueError("Each control_points_list item must be a tuple (cx, cy, ch)")
        cx, cy, ch = src
        cx = np.asarray(cx)
        cy = np.asarray(cy)
        ch = np.asarray(ch)
        if not (len(cx) == len(cy) == len(ch)):
            raise ValueError("Control point arrays must have the same length for each source")
        arrays_x.append(cx)
        arrays_y.append(cy)
        arrays_h.append(ch)

    # Concatenate (handle the case where there may be zero-length arrays)
    try:
        all_x = np.concatenate(arrays_x) if arrays_x else np.array([])
        all_y = np.concatenate(arrays_y) if arrays_y else np.array([])
        all_h = np.concatenate(arrays_h) if arrays_h else np.array([])
    except ValueError as exc:
        logger.error("Failed concatenating arrays: %s", exc)
        raise

    if not (len(all_x) == len(all_y) == len(all_h)):
        logger.error("Merged arrays length mismatch: %d, %d, %d", len(all_x), len(all_y), len(all_h))
        raise ValueError("Merged coordinate/value arrays have mismatched lengths")

    min_dist = config.get("min_separation_distance", 0.0) if isinstance(config, dict) else 0.0

    logger.info("Merged data points: %d. Cleaning duplicates with dist=%s...", len(all_x), min_dist)

    clean_x, clean_y, clean_h = remove_duplicate_points(all_x, all_y, all_h, min_dist)

    logger.info("Final data points after cleaning: %d", len(clean_x))

    return clean_x, clean_y, clean_h
