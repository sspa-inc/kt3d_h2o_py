from __future__ import annotations
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

def compute_resc(covmax: float, x: np.ndarray, y: np.ndarray, variogram_range: float) -> float:
    """
    Computes the rescaling factor for drift terms.
    
    Includes a safety floor to prevent instability when data points are
    clustered closely compared to the variogram range.
    """
    # Calculate squared radius from center
    x_center = np.mean(x)
    y_center = np.mean(y)
    radsqd = np.max((x - x_center)**2 + (y - y_center)**2)
    
    # Apply safety floor: radsqd = max(radsqd, variogram_range**2)
    # This prevents the rescaling factor from becoming excessively large
    # when the data domain is small relative to the correlation scale.
    safe_radsqd = max(radsqd, variogram_range**2)
    
    if safe_radsqd > 0:
        resc = np.sqrt(covmax / safe_radsqd)
    else:
        resc = 1.0
        
    logger.info(f"compute_resc: radsqd={radsqd:.2e}, safe_radsqd={safe_radsqd:.2e}, resc={resc:.2e}")
    return resc

# Placeholder for compute_polynomial_drift, assuming it exists elsewhere and returns a tuple of (np.ndarray, list[str])
def compute_polynomial_drift(x, y, config, resc):
    """
    Computes polynomial drift terms based on configuration.
    
    Parameters
    ----------
    x, y : np.ndarray
        Coordinates.
    config : dict or list
        If dict, looks for 'drift_terms' key.
        If list, treats as the list of term names directly.
    resc : float
        Rescaling factor.
    """
    term_names_to_compute = []
    if isinstance(config, dict):
        drift_cfg = config.get("drift_terms", {})
        # Maintain a stable order
        for term in ["linear_x", "linear_y", "quadratic_x", "quadratic_y"]:
            if drift_cfg.get(term):
                term_names_to_compute.append(term)
    elif isinstance(config, list):
        term_names_to_compute = config
    else:
        logger.warning("compute_polynomial_drift: config is neither dict nor list: %s", type(config))
        return np.zeros((len(x), 0)), []

    term_names = []
    drift_matrix_list = []

    for term in term_names_to_compute:
        if term == "linear_x":
            term_names.append("linear_x")
            drift_matrix_list.append(resc * x)
        elif term == "linear_y":
            term_names.append("linear_y")
            drift_matrix_list.append(resc * y)
        elif term == "quadratic_x":
            term_names.append("quadratic_x")
            drift_matrix_list.append(resc * x**2)
        elif term == "quadratic_y":
            term_names.append("quadratic_y")
            drift_matrix_list.append(resc * y**2)

    if not drift_matrix_list:
        return np.zeros((len(x), 0)), []

    return np.array(drift_matrix_list).T, term_names

def compute_drift_at_points(x, y, term_names, resc):
    """
    Recomputes drift terms at a set of points (e.g., grid nodes).
    
    Parameters
    ----------
    x, y : np.ndarray
        Coordinates of points.
    term_names : list[str]
        List of drift term names to compute.
    resc : float
        Rescaling factor.
        
    Returns
    -------
    np.ndarray
        Drift matrix (N_points, N_terms).
    """
    logger.info("Computing drift at %d points with resc=%.4e", len(x), resc)
    # Pass term_names directly as the 'config' parameter to compute_polynomial_drift
    drift_matrix, computed_names = compute_polynomial_drift(x, y, term_names, resc)
    logger.info("Generated drift matrix with shape %s and terms: %s", drift_matrix.shape, computed_names)
    
    return drift_matrix, computed_names # <--- Return the tuple

def drift_diagnostics(drift_matrix: np.ndarray, term_names: list[str], variogram: Any = None) -> None:
    """
    Performs diagnostic checks on the drift computation.

    Includes:
    1. Drift Magnitude Check: Assesses the magnitude of each drift term relative
       to the variogram sill.

    Parameters
    ----------
    drift_matrix : np.ndarray
        The drift matrix computed for the kriging model.
    term_names : list[str]
        List of names for the drift terms, corresponding to columns in drift_matrix.
    variogram : Any, optional
        The variogram model instance with a 'sill' attribute. If None, this check is skipped.
    """
    logger.info("--- Drift Diagnostics ---")

    # 1. Drift Magnitude Check
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

    logger.info("--- Drift Diagnostics Complete ---")

def _verify_linear_term(
    drift_col: np.ndarray,
    indep_var: np.ndarray,
    term_name: str,
    resc: float
) -> str:
    """
    Verify that drift_col = m * indep_var + b where m ≈ resc.

    Pass Criteria:
    1. R² > 0.999 (perfect linear fit)
    2. |m - resc| / resc < 0.01 (slope within 1% of resc)
    """
    # Fit degree-1 polynomial
    try:
        poly1 = np.polyfit(indep_var, drift_col, 1)
        slope = poly1[0]
        intercept = poly1[1]

        # Compute R²
        y_pred = np.polyval(poly1, indep_var)
        ss_res = np.sum((drift_col - y_pred) ** 2)
        ss_tot = np.sum((drift_col - np.mean(drift_col)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Check 1: Shape (R² > 0.999)
        shape_pass = r2 > 0.999

        # Check 2: Scaling (slope ≈ resc within 1%)
        if resc != 0:
            scale_error = abs(slope - resc) / abs(resc)
        else:
            scale_error = abs(slope)
        scale_pass = scale_error < 0.01

        # Log detailed results
        logger.info(
            f"Term '{term_name}': R²={r2:.6f}, Slope={slope:.6e}, "
            f"Expected={resc:.6e}, Error={scale_error:.4%}"
        )

        if not shape_pass:
            logger.error(f"  FAIL: R² ({r2:.6f}) <= 0.999 (not a perfect line)")
        if not scale_pass:
            logger.error(f"  FAIL: Slope error ({scale_error:.4%}) >= 1%")

        return "PASS" if (shape_pass and scale_pass) else "FAIL"
    except Exception as e:
        logger.error(f"Error verifying linear term '{term_name}': {e}")
        return "ERROR"


def _verify_quadratic_term(
    drift_col: np.ndarray,
    indep_var: np.ndarray,
    term_name: str,
    resc: float
) -> str:
    """
    Verify that drift_col = A * indep_var² + B * indep_var + K where A ≈ resc.

    Pass Criteria:
    1. R² > 0.999 (perfect quadratic fit)
    2. |A - resc| / resc < 0.01 (curvature within 1% of resc)
    3. Vertex V = -B/(2A) should be outside [min(indep_var), max(indep_var)]
       (WARNING only, does not fail test)
    """
    # Fit degree-2 polynomial
    try:
        poly2 = np.polyfit(indep_var, drift_col, 2)
        A = poly2[0]  # Curvature coefficient
        B = poly2[1]  # Linear coefficient
        K = poly2[2]  # Constant

        # Compute R²
        y_pred = np.polyval(poly2, indep_var)
        ss_res = np.sum((drift_col - y_pred) ** 2)
        ss_tot = np.sum((drift_col - np.mean(drift_col)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Check 1: Shape (R² > 0.999)
        shape_pass = r2 > 0.999

        # Check 2: Scaling (curvature A ≈ resc within 1%)
        if resc != 0:
            scale_error = abs(A - resc) / abs(resc)
        else:
            scale_error = abs(A)
        scale_pass = scale_error < 0.01

        # Check 3: Vertex location (safety check, warning only)
        if A != 0:
            vertex = -B / (2.0 * A)
            x_min, x_max = np.min(indep_var), np.max(indep_var)
            vertex_in_domain = x_min < vertex < x_max

            if vertex_in_domain:
                logger.warning(
                    f"Term '{term_name}': Vertex at {vertex:.4f} is INSIDE domain "
                    f"[{x_min:.4f}, {x_max:.4f}]. Parabola may reverse trend (U-turn)."
                )

        # Log detailed results
        logger.info(
            f"Term '{term_name}': R²={r2:.6f}, Curvature={A:.6e}, "
            f"Expected={resc:.6e}, Error={scale_error:.4%}"
        )

        if not shape_pass:
            logger.error(f"  FAIL: R² ({r2:.6f}) <= 0.999 (not a perfect parabola)")
        if not scale_pass:
            logger.error(f"  FAIL: Curvature error ({scale_error:.4%}) >= 1%")

        return "PASS" if (shape_pass and scale_pass) else "FAIL"
    except Exception as e:
        logger.error(f"Error verifying quadratic term '{term_name}': {e}")
        return "ERROR"


def verify_drift_physics(
    drift_matrix: np.ndarray,
    term_names: list[str],
    all_x: np.ndarray,
    all_y: np.ndarray,
    resc: float
) -> dict[str, str]:
    """
    Mathematically verify that each drift column follows its theoretical equation.
    """
    # Validate inputs
    # Check if drift_matrix is empty or has no columns before accessing shape[1]
    if drift_matrix is None or drift_matrix.size == 0:
        logger.info("No drift terms to verify.")
        return {}

    if len(term_names) != drift_matrix.shape[1]:
        raise ValueError(f"term_names length ({len(term_names)}) != drift_matrix columns ({drift_matrix.shape[1]})")

    all_x = np.asarray(all_x).ravel()
    all_y = np.asarray(all_y).ravel()

    if all_x.size != drift_matrix.shape[0]:
        raise ValueError(f"Coordinate array size ({all_x.size}) != drift_matrix rows ({drift_matrix.shape[0]})")

    results = {}

    for col_idx, term_name in enumerate(term_names):
        drift_col = drift_matrix[:, col_idx]

        # Determine independent variable based on term name
        if "_x" in term_name:
            indep_var = all_x
        elif "_y" in term_name:
            indep_var = all_y
        else:
            logger.warning(f"Unknown term type: {term_name}. Skipping verification.")
            results[term_name] = "SKIP"
            continue

        # Perform verification based on term type
        if "linear" in term_name:
            results[term_name] = _verify_linear_term(
                drift_col, indep_var, term_name, resc
            )
        elif "quadratic" in term_name:
            results[term_name] = _verify_quadratic_term(
                drift_col, indep_var, term_name, resc
            )
        else:
            logger.warning(f"Unknown term category: {term_name}. Skipping verification.")
            results[term_name] = "SKIP"

    return results

# Add to __all__ export
__all__ = [
    "compute_resc",
    "compute_polynomial_drift",
    "compute_drift_at_points",
    "drift_diagnostics",
    "verify_drift_physics"  # ADD THIS
]
