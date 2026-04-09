import json
from pathlib import Path
import importlib.util

import numpy as np

# Load drift module by path
SPEC_PATH_DRIFT = Path(__file__).parent / "drift.py"
spec_drift = importlib.util.spec_from_file_location("v2_drift", str(SPEC_PATH_DRIFT))
drift = importlib.util.module_from_spec(spec_drift)
spec_drift.loader.exec_module(drift)

# Load variogram module by path (lowercase module and class)
SPEC_PATH_VAR = Path(__file__).parent / "variogram.py"
spec_var = importlib.util.spec_from_file_location("v2_variogram", str(SPEC_PATH_VAR))
var_mod = importlib.util.module_from_spec(spec_var)
spec_var.loader.exec_module(var_mod)


def test_variogram_integration(tmp_path):
    # Create a minimal config containing variogram and drift_terms
    cfg = {
        "variogram": {"model": "spherical", "sill": 1.0, "range": 100.0, "nugget": 0.1},
        "drift_terms": {"linear_x": True, "linear_y": True},
    }

    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))

    v = var_mod.variogram(config_path=str(p))

    # Synthetic coordinates
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 100, 10)

    resc = drift.compute_resc(v.sill, x, y, v.range_)
    assert resc > 0

    dm, names = drift.compute_polynomial_drift(x, y, cfg, resc)
    assert dm.shape == (10, 2)
    assert names == ["linear_x", "linear_y"]
