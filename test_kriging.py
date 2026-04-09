import importlib.util
from pathlib import Path
import numpy as np
import pytest
from unittest import mock

# Load target module by path
SPEC_PATH = Path(__file__).parent / "kriging.py"
spec = importlib.util.spec_from_file_location("v2_kriging", str(SPEC_PATH))
kriging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kriging)


def test_build_uk_model_with_drift():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    h = np.array([10.0, 11.0, 12.0])
    # simple drift matrix with one column
    drift_matrix = np.column_stack([(x * 0.1)])

    # Patch the import routine to avoid importing pykrige and substitute a mock class
    with mock.patch.object(kriging, "_ensure_pykrige_import", return_value=None):
        mock_cls = mock.MagicMock()
        kriging.UniversalKriging = mock_cls
        variogram = mock.Mock(
            sill=1.0, 
            range_=10.0, 
            nugget=0.0, 
            model='spherical',
            anisotropy_enabled=False
        )
        kriging.build_uk_model(x, y, h, drift_matrix, variogram)

        # Assert the UniversalKriging class was instantiated
        assert mock_cls.called
        # capture kwargs from the instantiation
        call_args = mock_cls.call_args
        assert 'drift_terms' in call_args.kwargs
        assert call_args.kwargs['drift_terms'] == ['specified']
        assert 'specified_drift' in call_args.kwargs
        specified = call_args.kwargs['specified_drift']
        assert isinstance(specified, list)
        assert len(specified) == 1


def test_build_uk_model_no_drift():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    h = np.array([5.0, 6.0])
    drift_matrix = np.zeros((2, 0))

    with mock.patch.object(kriging, "_ensure_pykrige_import", return_value=None):
        mock_cls = mock.MagicMock()
        kriging.UniversalKriging = mock_cls
        variogram = mock.Mock(
            sill=1.0, 
            range_=10.0, 
            nugget=0.0, 
            model='spherical',
            anisotropy_enabled=False
        )
        kriging.build_uk_model(x, y, h, drift_matrix, variogram)
        call_args = mock_cls.call_args
        # ensure drift_terms was not passed in kwargs
        assert 'drift_terms' not in call_args.kwargs


def test_predict_at_points_column_mismatch():
    # Construct a mock model that reports it was trained with specified drift of length 2
    class DummyModel:
        def __init__(self):
            self.drift_terms = ['specified']
            self.specified_drift = [np.array([1.0]), np.array([2.0])]
        def execute(self, *args, **kwargs):
            return (np.array([0.0]), np.array([0.1]))

    model = DummyModel()
    # provide only one prediction column -> should raise
    drift_pred = np.zeros((1, 1))
    with pytest.raises(ValueError):
        kriging.predict_at_points(model, np.array([0.0]), np.array([0.0]), drift_pred)


def test_output_ols_coeffs(caplog):
    caplog.clear()
    # simple linear relation h = 2*x + 3*y
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    drift_matrix = np.column_stack([x, y])
    all_h = 2.0 * x + 3.0 * y
    coeffs = kriging.output_drift_coefficients(all_h, drift_matrix, ['linear_x', 'linear_y'])
    assert coeffs is not None
    assert coeffs.shape[0] == 2


def test_predict_on_grid_shapes(monkeypatch):
    # small grid
    cfg = {"grid": {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 3.0, "resolution": 1.0}}

    # monkeypatch predict_at_points to return z and ss arrays of correct length
    def fake_predict(uk_model, x, y, drift_matrix_pred=None):
        x = np.asarray(x)
        return np.arange(x.size).astype(float), (np.arange(x.size) * 0.1).astype(float)

    monkeypatch.setattr(kriging, 'predict_at_points', fake_predict)

    # construct a dummy uk_model (not used by fake_predict)
    dummy = object()
    GX, GY, Z, SS = kriging.predict_on_grid(dummy, cfg, ['linear_x'], 1.0)
    # shapes should match meshgrid
    assert Z.shape == GX.shape
    assert SS.shape == GX.shape


def test_cross_validate_small(monkeypatch):
    # small synthetic dataset
    x = np.array([0.0, 10.0, 20.0, 30.0])
    y = np.array([0.0, 5.0, 10.0, 15.0])
    h = 10.0 + 0.1 * x + 0.05 * y
    cfg = {"drift_terms": {"linear_x": True, "linear_y": True}}
    variogram = mock.Mock(sill=1.0, range_=100.0, nugget=0.0, model='spherical')

    # patch build_uk_model to return a dummy model and predict_at_points to return simple values
    class DummyModel:
        def execute(self, *args, **kwargs):
            return (np.array([0.0]), np.array([0.01]))

    monkeypatch.setattr(kriging, 'build_uk_model', lambda *a, **k: DummyModel())
    monkeypatch.setattr(kriging, 'predict_at_points', lambda model, xx, yy, drift_matrix_pred=None: (np.array([0.0]), np.array([0.01])))

    res = kriging.cross_validate(x, y, h, cfg, variogram)
    assert 'rmse' in res and 'mae' in res
    assert isinstance(res['predictions'], np.ndarray)
