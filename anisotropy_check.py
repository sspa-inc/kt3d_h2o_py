# -*- coding: utf-8 -*-
"""
anisotropy_check.py - Visual Diagnostics for UK_SSPA v2

Usage:
    python anisotropy_check.py
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- IMPORT FIX ---
# Try importing as a package first, then fallback to local
try:
    # UPDATED: lowercase module 'variogram' and class 'variogram'
    from v2_Code.variogram import variogram
    from v2_Code.drift import compute_resc, compute_polynomial_drift
    from v2_Code.kriging import output_drift_coefficients 
    from v2_Code.data import load_config, load_observation_wells, prepare_data
    from v2_Code.transform import get_transform_params, apply_transform
except ImportError:
    logger.info("Package import failed. Switching to local imports.")
    try:
        # UPDATED: lowercase module 'variogram' and class 'variogram'
        from variogram import variogram
        from drift import compute_resc, compute_polynomial_drift
        from kriging import output_drift_coefficients
        from data import load_config, load_observation_wells, prepare_data
        from transform import get_transform_params, apply_transform
    except ImportError as e:
        logger.error(f"Critical Import Error: {e}")
        sys.exit(1)

def generate_variogram_ellipse(center_x, center_y, range_major, ratio, angle_deg, n_points=100):
    """
    Generates the polygon coordinates for the variogram range ellipse.
    """
    # 1. Create a unit circle
    theta = np.linspace(0, 2*np.pi, n_points)
    x_circ = np.cos(theta)
    y_circ = np.sin(theta)
    
    # 2. Scale to Model Dimensions (Un-rotated)
    # Major axis (X) = range
    # Minor axis (Y) = range * ratio
    x_scaled = x_circ * range_major
    y_scaled = y_circ * (range_major * ratio)
    
    # 3. Rotate to Real-World Angle
    # Convert azimuth to arithmetic for plotting
    angle_rad = math.radians(90.0 - angle_deg)  # Convert azimuth to arithmetic for plotting
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    x_rot = x_scaled * cos_a - y_scaled * sin_a
    y_rot = x_scaled * sin_a + y_scaled * cos_a
    
    # 4. Translate to Center
    return x_rot + center_x, y_rot + center_y

def compute_directional_variograms(x, y, z, lags, tolerance=22.5):
    """
    Computes experimental semivariance in 4 primary directions.
    """
    from scipy.spatial.distance import pdist, squareform
    
    n = len(x)
    coords = np.column_stack((x, y))
    
    # Pairwise distances
    dist_matrix = squareform(pdist(coords))
    
    # Avoid division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        
        # Calculate angles (0 to 180 degrees)
        angles = np.degrees(np.arctan2(dy, dx))
        angles[angles < 0] += 180
    
    directions = {
        "0 (E-W)": 0.0,
        "45 (NE-SW)": 45.0,
        "90 (N-S)": 90.0,
        "135 (NW-SE)": 135.0
    }
    
    results = {}
    
    for label, target_angle in directions.items():
        # Mask pairs that align with this direction (+/- tolerance)
        angle_diff = np.abs(angles - target_angle)
        mask = (angle_diff <= tolerance) | (np.abs(angle_diff - 180) <= tolerance)
        
        gammas = []
        
        # Safe lag tolerance calculation
        if len(lags) > 1:
            step = lags[1] - lags[0]
            lag_tol = step / 2.0
        elif len(lags) == 1:
            lag_tol = lags[0] * 0.1
        else:
            lag_tol = 0.0

        for lag in lags:
            dist_mask = (dist_matrix >= lag - lag_tol) & (dist_matrix < lag + lag_tol)
            final_mask = mask & dist_mask
            
            # Get values for pairs (Upper triangle only)
            rows, cols = np.where(np.triu(final_mask, k=1))
            
            if len(rows) > 0:
                dz = z[rows] - z[cols]
                gamma = 0.5 * np.mean(dz**2)
                gammas.append(gamma)
            else:
                gammas.append(np.nan)
        results[label] = gammas
        
    return results

def main():
    try:
        # 1. Load Everything
        config_path = "config.json"
        if not os.path.exists(config_path):
            parent_config = os.path.join("..", "config.json")
            if os.path.exists(parent_config):
                config_path = parent_config
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, "config.json")
        
        if not os.path.exists(config_path):
            logger.error(f"Could not find config.json. Looked in: {os.getcwd()} and parent.")
            sys.exit(1)

        logger.info(f"Loading config from: {config_path}")
        config = load_config(config_path)
        
        # Instantiate the variogram class (renamed to var_model to avoid conflict)
        var_model = variogram(config=config)
        
        wx, wy, wh = load_observation_wells(config)
        all_x, all_y, all_h = prepare_data(wx, wy, wh, [], config)
        
        if len(all_x) == 0:
            logger.error("No data points loaded.")
            sys.exit(1)

        # 2. Calculate Residuals (Remove Drift)
        covmax = var_model.sill
        v_range = var_model.range_
        resc = compute_resc(covmax, all_x, all_y, v_range)
        drift_matrix, term_names = compute_polynomial_drift(all_x, all_y, config, resc)
        
        # OLS to get residuals
        coeffs = output_drift_coefficients(all_h, drift_matrix, term_names)
        if coeffs is not None:
            trend = drift_matrix @ coeffs
            residuals = all_h - trend
            logger.info("Drift removed. Analysing residuals.")
        else:
            residuals = all_h
            logger.info("No drift computed. Analysing raw values.")

        # 3. PLOT 1: The "Map" (Variogram Footprint)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Plot Data
        sc = ax.scatter(all_x, all_y, c=residuals, cmap='coolwarm', edgecolor='k', label='Residuals')
        plt.colorbar(sc, label='Residual Value')
        
        # Generate Ellipse
        center_x, center_y = np.mean(all_x), np.mean(all_y)
        ell_x, ell_y = generate_variogram_ellipse(
            center_x, center_y, 
            var_model.range_, 
            var_model.anisotropy_ratio, 
            var_model.angle_major
        )
        
        # Plot Ellipse
        ax.plot(ell_x, ell_y, 'r-', linewidth=3, label=f'Range Ellipse\nAngle={var_model.angle_major} deg')
        
        # Draw Axis Lines
        rad = math.radians(90.0 - var_model.angle_major)  # Convert azimuth to arithmetic for plotting
        dx = math.cos(rad) * var_model.range_
        dy = math.sin(rad) * var_model.range_
        ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy], 'r--', alpha=0.5)
        
        ax.set_title(f"Diagnostic 1: Variogram Footprint\nRange={var_model.range_}, Ratio={var_model.anisotropy_ratio}")
        ax.legend()
        ax.grid(True)
        
        plt.show()
        print("Check Plot 1: Does the red ellipse align with the 'streaks' in your data?")

        # 4. PLOT 2: The "Effect" (Transformed Isotropy Check)
        if var_model.anisotropy_enabled:
            params = get_transform_params(all_x, all_y, var_model.angle_major, var_model.anisotropy_ratio)
            tx, ty = apply_transform(all_x, all_y, params)
            title_prefix = "Transformed"
        else:
            tx, ty = all_x, all_y
            title_prefix = "Raw (No Transform)"
            
        lags = np.linspace(0, var_model.range_ * 1.5, 15)
        dir_results = compute_directional_variograms(tx, ty, residuals, lags)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for label, gammas in dir_results.items():
            ax2.plot(lags, gammas, marker='o', label=label)
            
        ax2.set_title(f"Diagnostic 2: Directional Variograms on {title_prefix} Coordinates")
        ax2.set_xlabel("Lag Distance")
        ax2.set_ylabel("Semivariance")
        ax2.legend()
        ax2.grid(True)
        
        plt.show()
        print("Check Plot 2: If working, all 4 lines should roughly overlap.")

    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()