
"""
Main module for UK_SSPA v2 - Micro-Task 4.2
Provides:
 - diagnose_kriging_system(uk_model, drift_matrix, term_names, variogram, all_h, label="")
 - generate_map(GX, GY, Z_grid, SS_grid, wx, wy, ctrl_points_list, config)

This module orchestrates the entire process, handles diagnostics, and generates outputs.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure the package directory is on sys.path for bare imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AEM_drift import compute_linesink_drift_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose_kriging_system(uk_model: Any, drift_matrix: np.ndarray, term_names: list[str], variogram: Any, all_h: np.ndarray, label: str = "") -> None:
    """
    Performs diagnostic checks on the kriging system.

    Includes:
    1. Exact Interpolation Test: Predicts at the first data point and compares
       with the actual observed value.
    2. Drift Magnitude Check: Assesses the magnitude of each drift term relative
       to the variogram sill.

    Parameters
    ----------
    uk_model : Any
        The UniversalKriging model instance (or a mock for testing).
    drift_matrix : np.ndarray
        The drift matrix used for training the kriging model.
    term_names : list[str]
        List of names for the drift terms, corresponding to columns in drift_matrix.
    variogram : Any
        The variogram model instance (or a mock for testing) with a 'sill' attribute.
    all_h : np.ndarray
        Array of all observed head values used for training.
    label : str, optional
        An optional label for the diagnostic output (e.g., "LOOCV"), by default "".
    """
    logger.info("--- Kriging System Diagnostics %s---", f"({label}) " if label else "")

    # 1. Exact Interpolation Test
    if all_h.size > 0:
        first_x = uk_model.X_ORIG[0]
        first_y = uk_model.Y_ORIG[0]
        actual_h = all_h[0]

        # Extract the first row of the drift matrix for prediction
        first_drift_row = drift_matrix[0:1, :] if drift_matrix.shape[1] > 0 else None

        try:
            # Use predict_at_points from kriging module
            try:
                from v2_Code.kriging import predict_at_points
            except ImportError:
                from kriging import predict_at_points
            pred_h, _ = predict_at_points(uk_model, np.array([first_x]), np.array([first_y]), first_drift_row)
            
            if pred_h.size > 0:
                error = abs(pred_h[0] - actual_h)
                logger.info("Exact Interpolation Test: Predicted at first point (%.2f, %.2f).", first_x, first_y)
                logger.info("  Actual: %.4f, Predicted: %.4f, Error: %.4e", actual_h, pred_h[0], error)
                if error > 0.01:
                    logger.warning("  Exact Interpolation Test WARNING: Error (%.4e) > 0.01. Model may not be interpolating exactly.", error)
            else:
                logger.warning("  Exact Interpolation Test WARNING: Prediction returned empty result.")
        except Exception as exc:
            logger.warning("  Exact Interpolation Test WARNING: Prediction failed: %s", exc)
    else:
        logger.info("Exact Interpolation Test skipped: No observation points available.")

    # 2. Drift Magnitude Check
    if drift_matrix.shape[1] > 0 and variogram is not None and hasattr(variogram, 'sill') and variogram.sill > 0:
        sill = float(variogram.sill)
        logger.info("Drift Magnitude Check (relative to variogram sill=%.4f):", sill)
        for i, term_name in enumerate(term_names):
            drift_column = drift_matrix[:, i]
            max_abs_drift = float(np.max(np.abs(drift_column)))
            ratio = max_abs_drift / sill
            logger.info("  Term '%s': Max Abs Drift=%.4e, Ratio to Sill=%.4e", term_name, max_abs_drift, ratio)
            if ratio > 1000.0:
                logger.warning("  Drift Magnitude Check WARNING: Term '%s' ratio (%.4e) to sill is very high (>1000).", term_name, ratio)
    else:
        logger.info("Drift Magnitude Check skipped: No drift terms or variogram sill not available/positive.")

    logger.info("--- Kriging System Diagnostics %sComplete ---", f"({label}) " if label else "")


def export_contours(GX: np.ndarray, GY: np.ndarray, Z_grid: np.ndarray, interval: float, out_path: str, crs: Any = None) -> None:
    """
    Exports contour lines from a grid to a shapefile.

    Parameters
    ----------
    GX, GY : np.ndarray
        Meshgrid arrays for X and Y coordinates.
    Z_grid : np.ndarray
        Grid of values to contour.
    interval : float
        Contour interval (must be > 0).
    out_path : str
        Path to save the output shapefile.
    crs : Any, optional
        Coordinate Reference System for the output shapefile.
    """
    if interval <= 0:
        raise ValueError(f"Contour interval must be > 0, got {interval}")

    
    # Use matplotlib to generate contour paths without plotting
    # We need to determine levels based on Z_grid range and interval
    z_min = np.nanmin(Z_grid)
    z_max = np.nanmax(Z_grid)
    
    logger.info("Z_grid min: %.4f, max: %.4f", z_min, z_max)

    if np.isnan(z_min) or np.isnan(z_max):
        logger.warning("Z_grid contains only NaNs. No contours generated.")
        return

    # Align levels to multiples of interval
    start_level = np.floor(z_min / interval) * interval
    levels = np.arange(start_level, z_max + interval, interval)
    
    logger.info("Contour levels: min=%.4f, max=%.4f, count=%d", levels.min(), levels.max(), len(levels))

    if len(levels) == 0:
        logger.warning("No contour levels found in the range of Z_grid.")
        return

    # Use matplotlib to compute contour lines (off-screen)
    cs = plt.contour(GX, GY, Z_grid, levels=levels)

    geoms = []
    elevations = []
    # Prefer using `cs.allsegs` which contains raw coordinate arrays
    # for each contour level; this is robust across Matplotlib versions
    for lev, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if seg.shape[0] < 2:
                continue
            # Create 3D LineString by adding elevation (Z) coordinate
            # Note: The original code in UK_SSPA.py creates 3D LineStrings.
            # If 2D is desired, remove the np.full part.
            seg_3d = np.column_stack((seg, np.full(seg.shape[0], float(lev))))
            from shapely.geometry import LineString
            geom = LineString(seg_3d)
            geoms.append(geom)
            elevations.append(float(lev))

    # Clear the contour from the active figure
    plt.clf()

    if not geoms:
        logger.warning("No contour lines generated.")
        return

    import geopandas as gpd
    gdf = gpd.GeoDataFrame({'elevation': elevations, 'geometry': geoms}, crs=crs)
    
    output_dir = os.path.dirname(out_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gdf.to_file(out_path)
    

def generate_map(GX: np.ndarray, GY: np.ndarray, Z_grid: np.ndarray, SS_grid: np.ndarray, wx: np.ndarray, wy: np.ndarray, ctrl_points_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None, config: dict) -> None:
    """
    Generates and displays/saves a map of kriged head and kriging standard deviation.

    Parameters
    ----------
    GX, GY : np.ndarray
        Meshgrid arrays for X and Y coordinates of the prediction grid.
    Z_grid : np.ndarray
        Kriged head values on the prediction grid.
    SS_grid : np.ndarray
        Kriging variance values on the prediction grid.
    wx, wy : np.ndarray
        X and Y coordinates of observation wells.
    ctrl_points_list : list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None
        List of control points (x, y, h) from line features.
    config : dict
        Configuration dictionary containing output settings (e.g., save_plots, plot_output_path).
    """
    logger.info("Generating kriging map...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot Head (ax1)
    head_contour = ax1.contourf(GX, GY, Z_grid, cmap='viridis', levels=20)
    fig.colorbar(head_contour, ax=ax1, label='Head')
    ax1.set_title('Kriged Head')
    ax1.scatter(wx, wy, color='red', marker='o', s=50, label='Observation Wells')

    if ctrl_points_list:
        # Flatten control points for plotting
        all_cx = np.concatenate([cp[0] for cp in ctrl_points_list])
        all_cy = np.concatenate([cp[1] for cp in ctrl_points_list])
        ax1.scatter(all_cx, all_cy, color='blue', marker='x', s=50, label='Control Points')
    ax1.legend()

    # Plot Variance (ax2)
    std_dev_contour = ax2.contourf(GX, GY, np.sqrt(SS_grid), cmap='magma', levels=20)
    fig.colorbar(std_dev_contour, ax=ax2, label='Std Deviation')
    ax2.set_title('Kriging Error (Std Dev)')

    plt.tight_layout()

    output_config = config.get('output', {})
    save_plots = output_config.get('save_plots', False)
    plot_output_path = output_config.get('plot_output_path', 'kriging_map.png')

    if save_plots:
        output_dir = os.path.dirname(plot_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(plot_output_path)
        logger.info("Kriging map saved to %s", plot_output_path)
    else:
        plt.show()
        logger.info("Kriging map displayed.")

    plt.close(fig)


def export_aux_points(x: np.ndarray, y: np.ndarray, h: np.ndarray, out_path: str, crs: Any = None) -> None:
    """
    Exports observation points to a shapefile.

    Parameters
    ----------
    x, y : np.ndarray
        X and Y coordinates of the points.
    h : np.ndarray
        Head values at the points.
    out_path : str
        Path to save the output shapefile.
    crs : Any, optional
        Coordinate Reference System for the output shapefile.
    """
    logger.info("Exporting auxiliary points to %s...", out_path)

    import geopandas as gpd
    from shapely.geometry import Point

    # Create Point geometries
    geoms = [Point(xi, yi) for xi, yi in zip(x, y)]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'head': h, 'geometry': geoms}, crs=crs)

    # Ensure output directory exists
    output_dir = os.path.dirname(out_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save to file
    gdf.to_file(out_path)
    logger.info("Successfully exported %d points to %s", len(gdf), out_path)


def main() -> None:
    """
    Main orchestration logic for the Universal Kriging workflow.
    """
    
    # Add project root to sys.path to allow importing v2_Code as a package
    # This fixes relative imports in submodules (e.g. kriging.py importing .drift)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from v2_Code.data import load_config, load_observation_wells, load_line_features, prepare_data
        from v2_Code.variogram import variogram
        from v2_Code.drift import compute_resc, compute_polynomial_drift, drift_diagnostics, verify_drift_physics
        from v2_Code.kriging import build_uk_model, output_drift_coefficients, cross_validate, predict_on_grid
        from v2_Code.transform import get_transform_params, apply_transform
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)

    try:
        # 1. Configuration
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if not os.path.exists(config_path):
            # Fallback for different execution contexts
            config_path = "config.json"
            
        config = load_config(config_path)
        variogram = variogram(config=config)
        logger.info("Configuration and Variogram initialized.")

        # 2. Data Loading
        wx, wy, wh = load_observation_wells(config)
        
        ctrl_points_list = []
        data_sources = config.get("data_sources", {})
        for source_name, source_conf in data_sources.items():
            if source_name == "observation_wells":
                continue
            
            # Check for toggle flag (default to True if not specified)
            if not source_conf.get("add_control_points", True):
                logger.info("Skipping data source '%s' (add_control_points=False)", source_name)
                continue

            # Assume other sources are line features for now (Phase 1)
            try:
                # NEW: Unpack the 4th value (cn for 'control point nugget')
                cx, cy, ch, cn = load_line_features(source_conf, config)
                
                if cx.size > 0:
                    # Pass only the first 3 to prepare_data (as it expects 3-tuples)
                    ctrl_points_list.append((cx, cy, ch))
            except Exception as e:
                logger.warning("Failed to load line feature source '%s': %s", source_name, e)


        # 3. Data Preparation - THIS IS WHERE THE OBS DATA TO BE INLCUDED IN THE KRIGING IS SPECIFIED
        all_x, all_y, all_h = prepare_data(wx, wy, wh, ctrl_points_list, config)

        if all_x.size == 0:
            logger.error("No data points available for kriging. Exiting.")
            return

        # 4. Coordinate Transformation (Anisotropy)
        transform_params = None
        x_model, y_model = all_x, all_y
        
        if variogram.anisotropy_enabled:
            logger.info("Anisotropy enabled. Computing coordinate transformation...")
            transform_params = get_transform_params(
                all_x, all_y,
                variogram.angle_major,
                variogram.anisotropy_ratio
            )
            x_model, y_model = apply_transform(all_x, all_y, transform_params)
            
            # Create a clone of variogram with anisotropy disabled for PyKrige
            # because we are handling it via pre-transformation
            variogram_for_kriging = variogram.clone()
            variogram_for_kriging.anisotropy_enabled = False
        else:
            variogram_for_kriging = variogram

        # 5. Drift Computation (on Model Coordinates)
        # Existing logic for linear/quadratic terms [cite: 347, 504-516]
        covmax = variogram.sill
        v_range = variogram.range_
        resc = compute_resc(covmax, x_model, y_model, v_range)
        # Initialize empty containers to avoid NameErrors
        drift_matrix_poly = np.zeros((len(x_model), 0))
        term_names_poly = []
        # Only compute if terms are actually enabled in config
        if any(config.get("drift_terms", {}).values()):
            drift_matrix_poly, term_names_poly = compute_polynomial_drift(x_model, y_model, config, resc)

        # 6. Linesink AEM Drift Computation
        drift_matrix_aem = np.zeros((len(x_model), 0))
        term_names_aem = []

        # Get the source configuration
        linesink_source_config = config.get("data_sources", {}).get("linesink_river", {})
        
        # Get the drift term configuration (could be bool or dict)
        drift_term_config = config.get("drift_terms", {}).get("linesink_river", False)

        # Parse drift configuration
        use_linesink = False
        apply_anisotropy = True # Default to Pythonic/Physical approach
        
        if isinstance(drift_term_config, bool):
            use_linesink = drift_term_config
        elif isinstance(drift_term_config, dict):
            use_linesink = drift_term_config.get("use", False)
            apply_anisotropy = drift_term_config.get("apply_anisotropy", True)

        # Dictionary to hold the factors
        trained_scaling_factors = {}
        
        # Determine which coordinates to use for the Drift Calculation
        if apply_anisotropy:
            # If applying anisotropy to river, use transformed wells (x_model)
            calc_x, calc_y = x_model, y_model
        else:
            # If river is kept raw (Physical space), use RAW wells (all_x)
            calc_x, calc_y = all_x, all_y

        # Check if enabled AND we have a valid path
        if use_linesink and linesink_source_config:
            import geopandas as gpd
            
            linesinks_path = linesink_source_config.get("path")
            if linesinks_path and os.path.exists(linesinks_path):
                linesinks_gdf = gpd.read_file(linesinks_path)
                
                # Retrieve the method from config
                method = linesink_source_config.get("rescaling_method", "adaptive")

                # drift_matrix_aem, term_names_aem, scaling_factors = compute_linesink_drift_matrix(
                #     calc_x, 
                #     calc_y, 
                #     linesinks_gdf, 
                #     linesink_source_config.get("group_column", "DriftTerm"), 
                #     transform_params, 
                #     variogram.sill,
                #     strength_col=linesink_source_config.get("strength_col", "resistance"),
                #     rescaling_method=method,
                #     apply_anisotropy=apply_anisotropy  # <--- PASS THE FLAG
                # )
                # Capture the 3rd return value: trained_scaling_factors
                drift_matrix_aem, term_names_aem, trained_scaling_factors = compute_linesink_drift_matrix(
                    calc_x, 
                    calc_y, 
                    linesinks_gdf, 
                    linesink_source_config.get("group_column", "DriftTerm"), 
                    transform_params, 
                    variogram.sill,
                    strength_col=linesink_source_config.get("strength_col", "resistance"),
                    rescaling_method=linesink_source_config.get("rescaling_method", "adaptive"),
                    apply_anisotropy=apply_anisotropy
                )                
                logger.info(f"Computed Linesink Drift. Anisotropy Applied: {apply_anisotropy}")
            else:
                logger.warning("Linesink drift enabled but shapefile path is missing or invalid.")
        

        # 7. Final Merge
        drift_matrix = np.hstack([drift_matrix_poly, drift_matrix_aem])
        term_names = term_names_poly + term_names_aem

        # 8. Build and Train Model
        # The build_uk_model function will now receive the expanded matrix [cite: 347, 491]
        uk_model = build_uk_model(x_model, y_model, all_h, drift_matrix, variogram_for_kriging)

        # 6. Drift Diagnostics
        drift_diagnostics(drift_matrix, term_names, variogram_for_kriging)

        # 6.5 Drift Physics Verification
        physics_results = verify_drift_physics(drift_matrix, term_names, x_model, y_model, resc)

        # Log summary
        if physics_results:
            passed = sum(1 for v in physics_results.values() if v == "PASS")
            total = len(physics_results)
            logger.info(f"Drift Physics Verification: {passed}/{total} terms PASSED")
            
            # Optional: Fail fast if any term fails
            failed_terms = [k for k, v in physics_results.items() if v == "FAIL"]
            if failed_terms:
                logger.error(f"CRITICAL: Drift verification FAILED for terms: {failed_terms}")
                logger.error("This indicates a mathematical error in drift computation.")
                # Uncomment to enforce strict failure:
                # sys.exit(1)
        else:
            logger.info("No drift terms to verify.")

        # 7. Build Model
        uk_model = build_uk_model(x_model, y_model, all_h, drift_matrix, variogram_for_kriging)

        # 8. System Diagnostics
        diagnose_kriging_system(uk_model, drift_matrix, term_names, variogram_for_kriging, all_h)
        output_drift_coefficients(all_h, drift_matrix, term_names)

        # 9. Cross-Validation (Conditional)
        if config.get("cross_validation", {}).get("enabled", False):
            logger.info("Running Leave-One-Out Cross-Validation...")
            # Note: CV needs to use the same transformation logic
            cv_results = cross_validate(all_x, all_y, all_h, config, variogram)
            logger.info("CV Results: RMSE=%.4f, MAE=%.4f", cv_results['rmse'], cv_results['mae'])

        # 10. Grid Prediction
        # PASS the trained_scaling_factors to the prediction function
        GX, GY, Z_grid, SS_grid = predict_on_grid(
            uk_model, 
            config, 
            term_names, 
            resc, 
            transform_params, 
            scaling_factors=trained_scaling_factors # <--- New argument
        )
        
        # 10. Outputs
        output_config = config.get("output", {})
        
        # Map Generation
        if output_config.get("generate_map", True):
            generate_map(GX, GY, Z_grid, SS_grid, wx, wy, ctrl_points_list, config)

        # Contour Export
        if output_config.get("export_contours", False):
            interval = output_config.get("contour_interval", 1.0)
            out_path = output_config.get("contour_output_path", "contours.shp")
            export_contours(GX, GY, Z_grid, interval, out_path)

        # Auxiliary Point Export
        if output_config.get("export_points", False):
            out_path = output_config.get("points_output_path", "observation_points.shp")
            export_aux_points(all_x, all_y, all_h, out_path)

        logger.info("Universal Kriging workflow completed successfully.")

    except Exception as exc:
        logger.error(f"An error occurred during the main execution loop: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["diagnose_kriging_system", "generate_map", "export_contours", "export_aux_points", "main", "verify_drift_physics"]
