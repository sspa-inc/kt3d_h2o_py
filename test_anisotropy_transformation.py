import numpy as np
import pytest
from v2_Code.transform import get_transform_params, apply_transform, invert_transform_coords
from v2_Code.drift import compute_resc
from v2_Code.variogram import variogram as Variogram
from v2_Code.kriging import build_uk_model, predict_at_points
from unittest.mock import MagicMock, patch

def test_coordinate_transformation_logic():
    # Test data
    x = np.array([0, 100, 0, 100])
    y = np.array([0, 0, 100, 100])
    angle = 45  # 45° azimuth (NE) — same physical direction in both conventions
    ratio = 0.5
    
    params = get_transform_params(x, y, angle, ratio)
    
    assert "center" in params
    assert "R" in params
    assert "S" in params
    assert np.allclose(params["center"], [50, 50])
    
    # Transform
    xt, yt = apply_transform(x, y, params)
    
    # Invert
    xi, yi = invert_transform_coords(xt, yt, params)
    
    assert np.allclose(x, xi)
    assert np.allclose(y, yi)

def test_compute_resc_safety_floor():
    # Small domain relative to range
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    v_range = 100.0
    sill = 1.0
    
    # Without safety floor, radsqd would be ~0.5
    # With safety floor, safe_radsqd should be 100^2 = 10000
    resc = compute_resc(sill, x, y, v_range)
    
    expected_resc = np.sqrt(1.0 / 10000.0)
    assert np.isclose(resc, expected_resc)

@patch('v2_Code.kriging.UniversalKriging')
def test_anisotropy_disabled_in_pykrige(mock_uk):
    # Mock variogram with anisotropy enabled
    variogram = MagicMock()
    variogram.anisotropy_enabled = True
    variogram.anisotropy_ratio = 0.5
    variogram.angle_major = 45  # azimuth convention (CW from North)
    variogram.sill = 1.0
    variogram.range_ = 100.0
    variogram.nugget = 0.0
    variogram.model = "spherical"
    
    # In main.py, we clone and disable
    variogram_for_kriging = MagicMock()
    variogram_for_kriging.anisotropy_enabled = False
    variogram_for_kriging.sill = 1.0
    variogram_for_kriging.range_ = 100.0
    variogram_for_kriging.nugget = 0.0
    variogram_for_kriging.model = "spherical"
    
    x = np.array([0, 1])
    y = np.array([0, 1])
    h = np.array([10, 20])
    
    build_uk_model(x, y, h, None, variogram_for_kriging)
    
    # Check that anisotropy_scaling was set to 1.0 in kwargs
    args, kwargs = mock_uk.call_args
    assert kwargs["anisotropy_scaling"] == 1.0
    assert kwargs["anisotropy_angle"] == 0.0

def test_azimuth_convention_major_axis_alignment():
    """Verify that angle_major=0 (azimuth=North) maps the North direction onto X-axis."""
    x = np.array([0, 0, 0])
    y = np.array([-50, 0, 50])  # Points along North-South (Y-axis)
    angle = 0.0  # Azimuth 0° = North = +Y direction
    ratio = 0.5
    
    params = get_transform_params(x, y, angle, ratio)
    xt, yt = apply_transform(x, y, params)
    
    # After transform: North (+Y) should map to X-axis
    # The centered coords are [0,-50], [0,0], [0,50]
    # With azimuth 0° (arithmetic 90°), rotation maps +Y onto +X
    # So xt should vary, yt_centered should be ~0 (before scaling)
    # Since ratio=0.5, Y gets scaled by 1/0.5=2, but centered Y is 0
    
    # The key check: points that were spread along Y should now be spread along X
    assert np.std(xt) > np.std(yt) * 0.1, "North direction should map to X-axis spread"
    
    # Roundtrip still works
    xi, yi = invert_transform_coords(xt, yt, params)
    assert np.allclose(x, xi, atol=1e-10)
    assert np.allclose(y, yi, atol=1e-10)

if __name__ == "__main__":
    pytest.main([__file__])
