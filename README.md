# UK_SSPA v2 — Universal Kriging for Water Level Mapping

![Tests](https://github.com/sspa-inc/kt3d_h2o_py/actions/workflows/tests.yml/badge.svg)

📖 **[Full Documentation](https://sspa-inc.github.io/kt3d_h2o_py/)**

---

## 🔬 Validation & Verification

**This software has been formally validated and verified.** The complete V&V report documents 10 validation tests covering all core modules including variogram models, drift terms, anisotropy transformations, and kriging accuracy. **Includes direct comparison against PyKrige** to verify wrapper equivalence and anisotropy consistency.

📥 **[Download V&V Report (PDF)](docs/validation/vv_report/_output/vv_report.pdf)**
🔍 **[Browse V&V Scripts & Results](docs/validation/)**

---

## What It Does
UK_SSPA v2 (Universal Kriging with Specified Spatial Polynomial and AEM drift) is a Python program for producing spatially interpolated groundwater level maps from point observation data. It implements Universal Kriging with user-specified drift terms, including polynomial trend functions and Analytic Element Method (AEM) linesink potentials derived from river geometry.

## Key Features
- Spherical, exponential, Gaussian, and linear variogram models
- Linear and quadratic polynomial drift terms
- Analytic Element Method (AEM) linesink drift for river influence
- Geometric anisotropy with coordinate pre-transformation
- Leave-one-out cross-validation (LOOCV)
- Grid prediction with contour export (shapefile)
- Config-driven pipeline — no code changes needed for different sites

## Quick Start
### 1. Install Dependencies

```bash
pip install numpy pandas geopandas shapely scipy matplotlib pykrige
```

Python 3.9 or later is recommended.

### 2. Prepare Your Input Shapefile

You need a **point shapefile** containing your observation well locations and a numeric water level column. The shapefile must be in a projected coordinate reference system (CRS) with linear units (e.g., metres or feet). The tool does **not** reproject data.

### 3. Create a Minimal `config.json`

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
    "points_output_path": "output/points.shp"
  },
  "cross_validation": {
    "enabled": false
  }
}
```

### 4. Run the Program

```bash
python main.py
```

By default, `main.py` looks for `config.json` in the current directory. To specify a different config file:

```bash
python main.py --config path/to/my_config.json
```

## Documentation
- [Overview](docs/overview.md)
- [Quick Start](docs/quickstart.md)
- [Configuration Reference](docs/configuration.md)
- [Workflow Reference](docs/workflow.md)
- [Data Contracts](docs/data-contracts.md)
- [Glossary](docs/glossary.md)

### Verification & Validation
- 📥 **[Download V&V Report (PDF)](docs/validation/vv_report/_output/vv_report.pdf)** — Formal validation report covering 10 tests
- [Tested Behaviors](docs/validation/tested-behaviors.md) — Summary of validation test cases
- [V&V Scripts](docs/validation/) — Browse all validation scripts and results

### Theory
- [Variogram Models](docs/theory/variogram-models.md)
- [Anisotropy](docs/theory/anisotropy.md)
- [Polynomial Drift](docs/theory/polynomial-drift.md)
- [AEM Linesink Drift](docs/theory/aem-linesink.md)

### API Reference
- [data.py](docs/api/data.md)
- [variogram.py](docs/api/variogram.md)
- [transform.py](docs/api/transform.md)
- [drift.py](docs/api/drift.md)
- [AEM_drift.py](docs/api/aem_drift.md)
- [kriging.py](docs/api/kriging.md)

### Examples
- [Ordinary Kriging](docs/examples/ex_ordinary_kriging.md)
- [Linear Drift](docs/examples/ex_linear_drift.md)
- [Linesink Drift](docs/examples/ex_linesink_drift.md)
- [Anisotropy](docs/examples/ex_anisotropy.md)

## Dependencies
- numpy
- pandas
- geopandas
- shapely
- scipy
- matplotlib
- pykrige

## Running Tests
```
pytest
```

## License
[To be determined]
