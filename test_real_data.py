import pytest
import os
import sys
from pathlib import Path
import shutil

# Add the v2_Code directory to sys.path so we can import modules
sys.path.append(str(Path(__file__).parent))

import main
import data

@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "Test" / "Test01" / "config.json").exists(),
    reason="Test/Test01/config.json not found — skipping integration test"
)
def test_run_model_with_test01_data(tmp_path):
    """
    Integration test running the full model pipeline using data from Test/Test01.
    """
    # Define paths
    base_dir = Path(__file__).parent.parent
    test_data_dir = base_dir / "Test" / "Test01"
    config_path = test_data_dir / "config.json"
    
    # Verify files exist
    assert config_path.exists(), f"Config file not found at {config_path}"
    assert (test_data_dir / "points.shp").exists()
    assert (test_data_dir / "linedrift.shp").exists()

    # Create a temporary output directory for this test run
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Load the original config
    config = data.load_config(str(config_path))
    
    # Update config to use absolute paths for input files (since we might be running from anywhere)
    # and set output path to tmp_path
    config["data_sources"]["observation_wells"]["path"] = str(test_data_dir / "points.shp")
    config["data_sources"]["river"]["path"] = str(test_data_dir / "linedrift.shp")
    
    # Ensure output section exists and configure it
    if "output" not in config:
        config["output"] = {}
    
    config["output"]["generate_map"] = True
    config["output"]["plot_output_path"] = str(output_dir / "final_map.png")
    config["output"]["contours_path"] = str(output_dir / "contours.shp")
    config["output"]["kriging_mean_path"] = str(output_dir / "kriging_mean.csv")
    config["output"]["kriging_variance_path"] = str(output_dir / "kriging_variance.csv")
    
    # We need to mock sys.argv or modify main to accept config dict/path
    # Since main.main() loads config internally from a hardcoded path or arg, 
    # let's look at main.py again. 
    # It seems main.py might not accept a config object directly in main().
    # Let's check main.py content first.
    
    # Actually, looking at previous file reads, main.py likely has a main() function.
    # Let's try to run the pipeline steps manually to avoid argument parsing issues
    # and to inject our modified config.
    
    # 1. Load Data
    print("Loading data...")
    wx, wy, wh = data.load_observation_wells(config)
    line_features = []
    if "river" in config["data_sources"]:
        cx, cy, ch = data.load_line_features(config["data_sources"]["river"], config)
        line_features.append((cx, cy, ch))
        
    # 2. Prepare Data
    print("Preparing data...")
    X, Y, H = data.prepare_data(wx, wy, wh, line_features, config)
    
    # 3. Drift
    print("Computing drift...")
    import drift
    # Calculate resc
    variogram_params = config["variogram"]
    sill = variogram_params["sill"]
    v_range = variogram_params["range"]
    
    resc = drift.compute_resc(sill, X, Y, v_range)
    drift_matrix, term_names = drift.compute_polynomial_drift(X, Y, config, resc)
    
    # 4. Kriging
    print("Building Kriging model...")
    import kriging
    from Variogram import Variogram
    
    # Create Variogram object (mocking the file load or creating directly if supported)
    # The Variogram class takes a config path. We can save our modified config to a temp file.
    temp_config_path = output_dir / "temp_config.json"
    import json
    with open(temp_config_path, 'w') as f:
        json.dump(config, f)
        
    variogram = Variogram(config_path=str(temp_config_path))
    
    uk_model = kriging.build_uk_model(X, Y, H, drift_matrix, variogram)
    
    # 5. Prediction
    print("Predicting on grid...")
    GX, GY, Z_grid, SS_grid = kriging.predict_on_grid(uk_model, config, term_names, resc)
    
    # 6. Output
    print("Generating outputs...")
    if config["output"]["generate_map"]:
        main.generate_map(GX, GY, Z_grid, SS_grid, wx, wy, line_features, config)
        assert Path(config["output"]["plot_output_path"]).exists()
        
    if "contours_path" in config["output"]:
        main.export_contours(GX, GY, Z_grid, 10.0, config["output"]["contours_path"]) # 10.0 is arbitrary interval
        assert Path(config["output"]["contours_path"]).exists()

    print("Test completed successfully.")
