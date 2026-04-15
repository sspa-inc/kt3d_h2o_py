import pytest
import numpy as np
import os
import sys
import geopandas as gpd
from pathlib import Path
import importlib.util
from unittest import mock
import json
import types

# Load the target module (v2_Code/main.py) by path
SPEC_PATH = Path(__file__).parent / "main.py"
spec = importlib.util.spec_from_file_location("v2_main", str(SPEC_PATH))
main_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_mod)

def test_diagnose_kriging_system_logging(caplog):
    caplog.set_level("INFO")
    
    # Mock uk_model
    uk_model = mock.Mock()
    uk_model.X_ORIG = np.array([100.0])
    uk_model.Y_ORIG = np.array([200.0])
    
    # Mock drift_matrix (1 point, 1 term)
    drift_matrix = np.array([[1.0]])
    term_names = ["linear_x"]
    
    # Mock variogram
    variogram = mock.Mock()
    variogram.sill = 1.0
    
    all_h = np.array([10.0])
    
    # Mock predict_at_points to avoid actual kriging
    # diagnose_kriging_system tries 'from v2_Code.kriging import predict_at_points' first
    with mock.patch("v2_Code.kriging.predict_at_points", return_value=(np.array([10.005]), np.array([0.1]))):
        main_mod.diagnose_kriging_system(uk_model, drift_matrix, term_names, variogram, all_h, label="Test")
    
    assert "--- Kriging System Diagnostics (Test) ---" in caplog.text
    assert "Exact Interpolation Test: Predicted at first point (100.00, 200.00)." in caplog.text
    assert "Actual: 10.0000, Predicted: 10.0050, Error: 5.0000e-03" in caplog.text
    assert "Term 'linear_x': Max Abs Drift=1.0000e+00, Ratio to Sill=1.0000e+00" in caplog.text

def test_diagnose_kriging_system_warning(caplog):
    caplog.set_level("WARNING")
    
    uk_model = mock.Mock()
    uk_model.X_ORIG = np.array([0.0])
    uk_model.Y_ORIG = np.array([0.0])
    
    drift_matrix = np.array([[1001.0]])
    term_names = ["linear_x"]
    variogram = mock.Mock()
    variogram.sill = 1.0
    all_h = np.array([10.0])
    
    # Mock prediction with high error
    with mock.patch("v2_Code.kriging.predict_at_points", return_value=(np.array([11.0]), np.array([0.1]))):
        main_mod.diagnose_kriging_system(uk_model, drift_matrix, term_names, variogram, all_h)
    
    assert "Exact Interpolation Test WARNING: Error (1.0000e+00) > 0.01" in caplog.text
    assert "Drift Magnitude Check WARNING: Term 'linear_x' ratio (1.0010e+03) to sill is very high (>1000)." in caplog.text

def test_export_contours_valid(tmp_path):
    # Create a simple grid
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    GX, GY = np.meshgrid(x, y)
    # Planar surface: Z = X + Y
    Z_grid = GX + GY
    
    out_path = tmp_path / "contours.shp"
    interval = 2.0
    
    main_mod.export_contours(GX, GY, Z_grid, interval, str(out_path))
    
    assert out_path.exists()
    gdf = gpd.read_file(str(out_path))
    assert not gdf.empty
    assert 'elevation' in gdf.columns
    # Z range is 0 to 20. With interval 2, we expect levels like 0, 2, 4, ..., 20
    unique_elevs = sorted(gdf['elevation'].unique())
    assert len(unique_elevs) > 1
    assert all(e % interval == 0 for e in unique_elevs)

def test_export_contours_invalid_interval():
    GX = np.zeros((2, 2))
    GY = np.zeros((2, 2))
    Z_grid = np.zeros((2, 2))
    
    with pytest.raises(ValueError, match="Contour interval must be > 0"):
        main_mod.export_contours(GX, GY, Z_grid, 0, "test.shp")

def test_export_contours_no_levels(caplog, tmp_path):
    # Flat surface, interval larger than range
    GX, GY = np.meshgrid([0, 1], [0, 1])
    Z_grid = np.zeros((2, 2))
    out_path = tmp_path / "no_levels.shp"
    
    Z_grid_nan = np.full((2, 2), np.nan)
    
    main_mod.export_contours(GX, GY, Z_grid_nan, 1.0, str(out_path))
    assert "Z_grid contains only NaNs" in caplog.text

def test_export_contours_creates_dir(tmp_path):
    GX, GY = np.meshgrid([0, 1], [0, 1])
    Z_grid = GX + GY
    nested_path = tmp_path / "subdir" / "contours.shp"
    
    main_mod.export_contours(GX, GY, Z_grid, 0.5, str(nested_path))
    assert nested_path.exists()

def test_export_aux_points_valid(tmp_path):
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 10.0, 20.0])
    h = np.array([100.0, 110.0, 120.0])
    out_path = tmp_path / "points.shp"
    
    main_mod.export_aux_points(x, y, h, str(out_path))
    
    assert out_path.exists()
    gdf = gpd.read_file(str(out_path))
    assert len(gdf) == 3
    assert 'head' in gdf.columns
    assert np.allclose(gdf['head'], h)
    # Check first point geometry
    assert gdf.geometry[0].x == 0.0
    assert gdf.geometry[0].y == 0.0

def test_export_aux_points_empty(tmp_path):
    x = np.array([])
    y = np.array([])
    h = np.array([])
    out_path = tmp_path / "empty_points.shp"
    
    main_mod.export_aux_points(x, y, h, str(out_path))
    
    assert out_path.exists()
    gdf = gpd.read_file(str(out_path))
    assert gdf.empty

@mock.patch("matplotlib.pyplot.savefig")
@mock.patch("matplotlib.pyplot.show")
@mock.patch("matplotlib.pyplot.subplots")
@mock.patch("matplotlib.pyplot.tight_layout")
@mock.patch("matplotlib.pyplot.close")
def test_generate_map_save(mock_close, mock_tight_layout, mock_subplots, mock_show, mock_savefig, tmp_path):
    # Mock subplots to return a figure and axes
    mock_fig = mock.Mock()
    mock_ax1 = mock.Mock()
    mock_ax2 = mock.Mock()
    mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

    GX, GY = np.meshgrid([0, 1], [0, 1])
    Z_grid = np.zeros((2, 2))
    SS_grid = np.ones((2, 2))
    wx, wy = np.array([0.5]), np.array([0.5])
    
    plot_path = tmp_path / "map.png"
    config = {
        "output": {
            "save_plots": True,
            "plot_output_path": str(plot_path)
        }
    }
    
    main_mod.generate_map(GX, GY, Z_grid, SS_grid, wx, wy, None, config)
    
    # Verify savefig was called with the correct path
    mock_savefig.assert_called_with(str(plot_path))

def test_export_water_level_ascii_grid_valid(tmp_path):
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 10.0, 20.0])
    GX, GY = np.meshgrid(x, y)
    Z_grid = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ])

    out_path = tmp_path / "water_levels.asc"
    main_mod.export_water_level_ascii_grid(GX, GY, Z_grid, str(out_path))

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8").splitlines()
    assert content[0] == "ncols         3"
    assert content[1] == "nrows         3"
    assert content[2] == "xllcorner     -5"
    assert content[3] == "yllcorner     -5"
    assert content[4] == "cellsize      10"
    assert content[5] == "NODATA_value  -9999"
    assert content[6].split() == ["7", "8", "9"]
    assert content[7].split() == ["4", "-9999", "6"]
    assert content[8].split() == ["1", "2", "3"]


def test_export_water_level_ascii_grid_requires_square_cells(tmp_path):
    GX, GY = np.meshgrid([0.0, 10.0, 20.0], [0.0, 20.0, 40.0])
    Z_grid = np.arange(9, dtype=float).reshape(3, 3)

    with pytest.raises(ValueError, match="square cells"):
        main_mod.export_water_level_ascii_grid(GX, GY, Z_grid, str(tmp_path / "bad.asc"))


def test_export_water_level_tif_valid(tmp_path, monkeypatch):
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 10.0, 20.0])
    GX, GY = np.meshgrid(x, y)
    Z_grid = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ])
    out_path = tmp_path / "water_levels.tif"

    captured = {}

    class DummyDataset:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data, band):
            captured["data"] = data.copy()
            captured["band"] = band

    def fake_open(path, mode, driver, height, width, count, dtype, nodata, transform):
        captured["path"] = path
        captured["mode"] = mode
        captured["driver"] = driver
        captured["height"] = height
        captured["width"] = width
        captured["count"] = count
        captured["dtype"] = dtype
        captured["nodata"] = nodata
        captured["transform"] = transform
        Path(path).write_bytes(b"FAKE_TIF")
        return DummyDataset()

    fake_rasterio = types.SimpleNamespace(open=fake_open)
    fake_transform_module = types.SimpleNamespace(from_origin=lambda x0, y0, dx, dy: (x0, y0, dx, dy))

    monkeypatch.setitem(sys.modules, "rasterio", fake_rasterio)
    monkeypatch.setitem(sys.modules, "rasterio.transform", fake_transform_module)

    main_mod.export_water_level_tif(GX, GY, Z_grid, str(out_path))

    assert out_path.exists()
    assert captured["driver"] == "GTiff"
    assert captured["height"] == 3
    assert captured["width"] == 3
    assert captured["count"] == 1
    assert captured["band"] == 1
    assert captured["transform"] == (-5.0, 25.0, 10.0, 10.0)
    np.testing.assert_allclose(
        captured["data"],
        np.array([
            [7.0, 8.0, 9.0],
            [4.0, -9999.0, 6.0],
            [1.0, 2.0, 3.0],
        ], dtype=np.float32),
    )


def test_export_water_level_tif_requires_rasterio(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "rasterio", raising=False)
    monkeypatch.delitem(sys.modules, "rasterio.transform", raising=False)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rasterio" or name.startswith("rasterio"):
            raise ImportError("No module named 'rasterio'")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="rasterio"):
            main_mod.export_water_level_tif(
                np.array([[0.0, 1.0], [0.0, 1.0]]),
                np.array([[0.0, 0.0], [1.0, 1.0]]),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                str(tmp_path / "missing.tif"),
            )


def test_main_pipeline_execution(tmp_path):
    # Since imports happen inside main(), we need to patch sys.modules or the source modules directly.
    # However, since we are running tests where v2_Code is in path, we can patch the source modules.
    
    # We need to make sure the modules are imported before we patch them
    # main() imports from v2_Code.* so we patch those paths
    import v2_Code.data
    import v2_Code.drift
    import v2_Code.kriging
    import v2_Code.variogram

    with mock.patch("v2_Code.data.load_config") as mock_load_config, \
         mock.patch("v2_Code.data.load_observation_wells") as mock_wells, \
         mock.patch("v2_Code.data.prepare_data") as mock_prepare, \
         mock.patch("v2_Code.drift.compute_resc") as mock_resc, \
         mock.patch("v2_Code.drift.compute_polynomial_drift") as mock_drift, \
         mock.patch("v2_Code.kriging.build_uk_model") as mock_build, \
         mock.patch("v2_Code.kriging.predict_on_grid") as mock_predict, \
         mock.patch("v2_Code.variogram.variogram") as mock_v_class, \
         mock.patch.object(main_mod, "export_water_level_tif") as mock_export_tif, \
         mock.patch.object(main_mod, "export_water_level_ascii_grid") as mock_export_asc, \
         mock.patch("os.path.exists", return_value=True):
        
        mock_load_config.return_value = {
            "data_sources": {"observation_wells": {}},
            "output": {
                "generate_map": False,
                "export_water_level_tif": True,
                "water_level_tif_output_path": str(tmp_path / "out.tif"),
                "export_water_level_asc": True,
                "water_level_asc_output_path": str(tmp_path / "out.asc"),
            }
        }
        mock_wells.return_value = (np.array([0]), np.array([0]), np.array([10]))
        mock_prepare.return_value = (np.array([0]), np.array([0]), np.array([10]))
        mock_resc.return_value = 1.0
        mock_drift.return_value = (np.array([[1]]), ["linear_x"])
        # Mock uk_model to have X_ORIG and Y_ORIG for diagnostics
        mock_uk_model = mock.Mock()
        mock_uk_model.X_ORIG = np.array([0.0])
        mock_uk_model.Y_ORIG = np.array([0.0])
        mock_build.return_value = mock_uk_model

        mock_predict.return_value = (np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
        
        v_inst = mock_v_class.return_value
        v_inst.sill = 1.0
        v_inst.range_ = 100.0
        v_inst.anisotropy_enabled = False
        
        # Run main
        # Also mock matplotlib to avoid TclError in generate_map
        with mock.patch("matplotlib.pyplot.subplots") as mock_subplots, \
             mock.patch("matplotlib.pyplot.show"), \
             mock.patch("matplotlib.pyplot.savefig"), \
             mock.patch("matplotlib.pyplot.tight_layout"), \
             mock.patch("matplotlib.pyplot.close"):
             
            mock_fig = mock.Mock()
            mock_ax1 = mock.Mock()
            mock_ax2 = mock.Mock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            
            main_mod.main()
        
        # Verify key steps were called
        mock_load_config.assert_called()
        mock_wells.assert_called()
        mock_prepare.assert_called()
        mock_build.assert_called()
        mock_predict.assert_called()
        mock_export_tif.assert_called_once()
        mock_export_asc.assert_called_once()
