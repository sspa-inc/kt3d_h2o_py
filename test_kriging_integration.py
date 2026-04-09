"""
Integration test for kriging module (Task 3.7).

Tests the full pipeline: data preparation -> drift computation -> model building ->
grid prediction -> LOOCV -> OLS diagnostics.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from unittest import mock

# Load modules by path
SPEC_PATH_DATA = Path(__file__).parent / "data.py"
spec_data = importlib.util.spec_from_file_location("v2_data", str(SPEC_PATH_DATA))
data = importlib.util.module_from_spec(spec_data)
spec_data.loader.exec_module(data)

SPEC_PATH_DRIFT = Path(__file__).parent / "drift.py"
spec_drift = importlib.util.spec_from_file_location("v2_drift", str(SPEC_PATH_DRIFT))
drift = importlib.util.module_from_spec(spec_drift)
spec_drift.loader.exec_module(drift)

SPEC_PATH_KRIGING = Path(__file__).parent / "kriging.py"
spec_kriging = importlib.util.spec_from_file_location("v2_kriging", str(SPEC_PATH_KRIGING))
kriging = importlib.util.module_from_spec(spec_kriging)
spec_kriging.loader.exec_module(kriging)


def test_kriging_integration_e2e():
    """
    End-to-end integration test:
    1. Create synthetic wells with planar trend
    2. Build config with drift terms
    3. Create variogram mock
    4. Compute resc and drift matrix
    5. Build UK model
    6. Predict on grid
    7. Run LOOCV
    8. Output OLS coefficients
    """
    # Step 1: Synthetic wells with planar trend h = 10 + 0.1*x + 0.05*y
    x_wells = np.array([0.0, 100.0, 0.0, 100.0])
    y_wells = np.array([0.0, 0.0, 100.0, 100.0])
    h_wells = 10.0 + 0.1 * x_wells + 0.05 * y_wells

    # Step 2: Config with drift terms
    config = {
        "drift_terms": {"linear_x": True, "linear_y": True},
        "grid": {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0, "resolution": 50.0},
    }

    # Step 3: Variogram mock
    variogram = mock.Mock(
        sill=1.0,
        range_=50.0,
        nugget=0.0,
        model='spherical',
        anisotropy_enabled=False
    )

    # Step 4: Compute resc and drift matrix
    covmax = variogram.sill
    variogram_range = variogram.range_
    resc = drift.compute_resc(covmax, x_wells, y_wells, variogram_range)
    assert resc > 0, "resc should be positive"

    drift_matrix, term_names = drift.compute_polynomial_drift(x_wells, y_wells, config, resc)
    assert drift_matrix.shape == (4, 2), "drift matrix should be 4x2"
    assert term_names == ["linear_x", "linear_y"], "term names should match config order"

    # Step 5: Build UK model (mock pykrige)
    with mock.patch.object(kriging, "_ensure_pykrige_import", return_value=None):
        mock_uk_cls = mock.MagicMock()
        kriging.UniversalKriging = mock_uk_cls
        
        uk_model = kriging.build_uk_model(x_wells, y_wells, h_wells, drift_matrix, variogram)
        assert mock_uk_cls.called, "UniversalKriging should be instantiated"
        
        # Verify drift_terms was set correctly
        call_args = mock_uk_cls.call_args
        assert call_args.kwargs.get('drift_terms') == ['specified']

    # Step 6: Predict on grid (mock predict_at_points)
    def fake_predict(uk_model, x, y, drift_matrix_pred=None):
        """Return synthetic predictions matching grid size."""
        x = np.asarray(x)
        return np.ones(x.size) * 15.0, np.ones(x.size) * 0.5

    with mock.patch.object(kriging, 'predict_at_points', side_effect=fake_predict):
        GX, GY, Z_grid, SS_grid = kriging.predict_on_grid(uk_model, config, term_names, resc)
        
        # Verify shapes
        assert GX.shape == GY.shape, "GX and GY should have same shape"
        assert Z_grid.shape == GX.shape, "Z_grid should match meshgrid shape"
        assert SS_grid.shape == GX.shape, "SS_grid should match meshgrid shape"
        assert Z_grid.size > 0, "Grid predictions should not be empty"

    # Step 7: Run LOOCV (mock build_uk_model and predict_at_points)
    class DummyModel:
        pass

    def fake_build(*args, **kwargs):
        return DummyModel()

    with mock.patch.object(kriging, 'build_uk_model', side_effect=fake_build):
        with mock.patch.object(kriging, 'predict_at_points', side_effect=fake_predict):
            cv_results = kriging.cross_validate(x_wells, y_wells, h_wells, config, variogram)
            
            # Verify results structure
            assert 'rmse' in cv_results
            assert 'mae' in cv_results
            assert 'q1' in cv_results
            assert 'q2' in cv_results
            assert 'predictions' in cv_results
            assert 'variances' in cv_results
            assert 'observations' in cv_results
            assert cv_results['predictions'].shape[0] == 4

    # Step 8: Output OLS coefficients
    coeffs = kriging.output_drift_coefficients(h_wells, drift_matrix, term_names)
    assert coeffs is not None, "OLS coefficients should be computed"
    assert coeffs.shape[0] == 2, "Should have 2 coefficients for 2 drift terms"


def test_kriging_integration_no_drift():
    """
    Integration test without drift terms.
    """
    x_wells = np.array([0.0, 100.0])
    y_wells = np.array([0.0, 100.0])
    h_wells = np.array([10.0, 20.0])

    config = {
        "drift_terms": {},  # No drift terms
        "grid": {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0, "resolution": 50.0},
    }

    variogram = mock.Mock(
        sill=1.0,
        range_=50.0,
        nugget=0.0,
        model='spherical',
        anisotropy_enabled=False
    )

    # Compute resc and drift (should be empty)
    resc = drift.compute_resc(variogram.sill, x_wells, y_wells, variogram.range_)
    drift_matrix, term_names = drift.compute_polynomial_drift(x_wells, y_wells, config, resc)
    
    assert drift_matrix.shape == (2, 0), "drift matrix should be empty (2x0)"
    assert term_names == [], "term names should be empty"

    # Build model without drift
    with mock.patch.object(kriging, "_ensure_pykrige_import", return_value=None):
        mock_uk_cls = mock.MagicMock()
        kriging.UniversalKriging = mock_uk_cls
        
        uk_model = kriging.build_uk_model(x_wells, y_wells, h_wells, drift_matrix, variogram)
        
        # Verify no drift_terms kwarg was passed
        call_args = mock_uk_cls.call_args
        assert 'drift_terms' not in call_args.kwargs, "drift_terms should not be passed for empty drift"
