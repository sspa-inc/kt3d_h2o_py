# Quickstart Guide

This guide walks you through running UK_SSPA v2 end-to-end in the simplest possible configuration: ordinary kriging with no drift terms and no anisotropy.

---

## 1. Install Dependencies

```bash
pip install numpy pandas geopandas shapely scipy matplotlib pykrige
```

Python 3.9 or later is recommended.

---

## 2. Prepare Your Input Shapefile

You need a **point shapefile** containing your observation well locations and a numeric water level column. The shapefile must be in a projected coordinate reference system (CRS) with linear units (e.g., metres or feet). The tool does **not** reproject data.

Example attribute table:

| FID | geometry (Point) | WaterLevel |
|-----|-----------------|------------|
| 0   | POINT(1000, 2000) | 12.4 |
| 1   | POINT(3500, 1800) | 11.9 |
| ... | ...             | ...  |

---

## 3. Create a Minimal `config.json`

Place this file in the same directory as `main.py`:

```json
{
  "data_sources": {
    "observation_wells": {
      "path": "path/to/your/wells.shp",
      "water_level_col": "WaterLevel"
    }
  },
  "variogram": {
    "model": "spherical",
    "sill": 1.0,
    "range": 1000.0,
    "nugget": 0.0,
    "anisotropy": {
      "enabled": false
    },
    "advanced": {
      "search_radius": null,
      "max_neighbors": null,
      "min_neighbors": null
    }
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
    "x_max": 5000.0,
    "y_min": 0.0,
    "y_max": 5000.0,
    "resolution": 100.0
  },
  "min_separation_distance": 1.0,
  "output": {
    "generate_map": true,
    "export_contours": false,
    "contour_interval": 1.0,
    "contour_output_path": "output/contours.shp",
    "export_points": false,
    "points_output_path": "output/points.shp",
    "export_water_level_tif": false,
    "water_level_tif_output_path": "output/water_levels.tif",
    "export_water_level_asc": false,
    "water_level_asc_output_path": "output/water_levels.asc"
  },
  "cross_validation": {
    "enabled": false
  }
}
```

**Key settings to adjust:**

| Setting | What to change |
|---|---|
| `data_sources.observation_wells.path` | Path to your point shapefile |
| `data_sources.observation_wells.water_level_col` | Name of the water level column in your shapefile |
| `variogram.sill` | Approximate variance of your water level data |
| `variogram.range` | Approximate distance (in CRS units) beyond which wells are uncorrelated |
| `grid.*` | Bounding box and resolution of the prediction grid (in CRS units) |

---

## 4. Run the Program

```bash
python main.py
```

By default, `main.py` looks for `config.json` in the current directory. To specify a different config file:

```bash
python main.py --config path/to/my_config.json
```

---

## 5. Expected Log Output

A successful run produces log messages at each pipeline stage. You should see milestones similar to the following:

```
INFO  Configuration loaded: model=spherical, sill=1.0, range=1000.0, nugget=0.0
INFO  Anisotropy disabled — using isotropic kriging
INFO  Loaded 42 observation wells from path/to/your/wells.shp
INFO  Data prepared: 42 points after duplicate removal
INFO  No drift terms enabled — running ordinary kriging
INFO  UK model built successfully
INFO  Predicting on grid: 50 x 50 = 2500 points
INFO  Prediction complete
INFO  Map generated
```

If any stage fails, an error message will identify the cause (e.g., missing column, invalid config value, singular kriging matrix).

---

## 6. Output Files

Depending on your `output` configuration:

| Output | Config key | Description |
|---|---|---|
| Map (displayed) | `generate_map: true` | Matplotlib figure showing the predicted water level surface |
| Contour shapefile | `export_contours: true` | LineString shapefile at `contour_output_path` |
| Auxiliary points | `export_points: true` | Point shapefile with x, y, predicted h at `points_output_path` |
| Water-level GeoTIFF | `export_water_level_tif: true` | Raster `.tif` at `water_level_tif_output_path` |
| Water-level ASCII grid | `export_water_level_asc: true` | Arc/Info ASCII Grid `.asc` at `water_level_asc_output_path` |

---

## Next Steps

Once ordinary kriging is working, explore more advanced features:

- **Add polynomial drift** (regional gradient): see [`docs/configuration.md`](configuration.md) → `drift_terms`
- **Add anisotropy** (directional correlation): see [`docs/theory/anisotropy.md`](theory/anisotropy.md)
- **Add AEM linesink drift** (river influence): see [`docs/theory/aem-linesink.md`](theory/aem-linesink.md)
- **Enable cross-validation**: set `cross_validation.enabled: true`
- **Full configuration reference**: [`docs/configuration.md`](configuration.md)
- **Full pipeline walkthrough**: [`docs/workflow.md`](workflow.md)
