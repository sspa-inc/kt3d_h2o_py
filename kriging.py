"""
Kriging utilities for UK_SSPA v2 - Micro-Task 3.1

Provides:
 - build_uk_model(all_x, all_y, all_h, drift_matrix, variogram) -> UniversalKriging

This module wraps pykrige.uk.UniversalKriging initialization to ensure
we always pass 'drift_terms=["specified"]' when providing specified drift
and to centralize anisotropy logging/handling (Phase 1: drift computed on
raw coords; PyKrige may apply anisotropy adjustments internally).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Lazy import placeholder for pykrige class - imported at call-time to allow
# unit tests to mock 'pykrige.uk.UniversalKriging' without requiring pykrige
# to be actually installed at import time.
UniversalKriging: Any = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _ensure_pykrige_import() -> None:
    """Attempt to import UniversalKriging if not already available.

    Raises ImportError with a friendlier message if pykrige is not installed.
    """
    global UniversalKriging
    if UniversalKriging is not None:
        return
    try:
        from pykrige.uk import UniversalKriging as _UK  # type: ignore

        UniversalKriging = _UK
    except Exception as exc:  # ImportError or others
        logger.error("pykrige is required for kriging but could not be imported: %s", exc)
        raise


def build_uk_model(all_x: np.ndarray, all_y: np.ndarray, all_h: np.ndarray, drift_matrix: np.ndarray | None, variogram: Any):
    """Construct and initialize a pykrige.uk.UniversalKriging model.

    Parameters
    ----------
    all_x, all_y, all_h : np.ndarray
        Training coordinates and values (1-D arrays).
    drift_matrix : np.ndarray or None
        Drift matrix computed for training points (shape N x n_terms) or None.
    variogram : object
        Variogram-like object with attributes: `sill`, `range_` (or `range`),
        `nugget`, `model`, and optional anisotropy fields `anisotropy_enabled`,
        `anisotropy_ratio`, `angle_major`.

    Returns
    -------
    uk_model : UniversalKriging instance
        Initialized PyKrige UniversalKriging object.

    Raises
    ------
    Exception
        Re-raises any exception from UniversalKriging initialization after
        logging context.
    """
    # Ensure pykrige class is available (test harness may mock it)
    _ensure_pykrige_import()

    # Coerce arrays
    x = np.asarray(all_x).ravel()
    y = np.asarray(all_y).ravel()
    h = np.asarray(all_h).ravel()

    # Prepare variogram parameters dict
    try:
        sill = float(getattr(variogram, "sill"))
    except Exception:
        sill = float(getattr(variogram, "sill", 0.0))
    try:
        rng = float(getattr(variogram, "range_", getattr(variogram, "range", 0.0)))
    except Exception:
        rng = float(getattr(variogram, "range_", getattr(variogram, "range", 0.0)))
    try:
        nugget = float(getattr(variogram, "nugget"))
    except Exception:
        nugget = float(getattr(variogram, "nugget", 0.0))

    variogram_parameters = {"sill": sill, "range": rng, "nugget": nugget}

    # Base kwargs for UniversalKriging
    uk_kwargs: dict[str, Any] = {
        "variogram_model": getattr(variogram, "model", "spherical"),
        "variogram_parameters": variogram_parameters,
        "verbose": False,
        "enable_plotting": False,
    }

    # Handle anisotropy
    anisotropy_enabled = bool(getattr(variogram, "anisotropy_enabled", False))
    if anisotropy_enabled:
        # PyKrige expects anisotropy_scaling (ratio) and anisotropy_angle (deg?)
        anisotropy_ratio = float(getattr(variogram, "anisotropy_ratio", 1.0))
        anisotropy_angle = float(getattr(variogram, "angle_major", 0.0))
        uk_kwargs["anisotropy_scaling"] = anisotropy_ratio
        uk_kwargs["anisotropy_angle"] = 90.0 - anisotropy_angle  # Convert azimuth to arithmetic for PyKrige
        logger.info("Anisotropy enabled in PyKrige: scaling=%s angle=%s", anisotropy_ratio, anisotropy_angle)
    else:
        # Explicitly disable to ensure no internal transformation if we pre-transformed
        uk_kwargs["anisotropy_scaling"] = 1.0
        uk_kwargs["anisotropy_angle"] = 0.0

    # Attach specified drift if provided
    if drift_matrix is not None and getattr(drift_matrix, "ndim", 1) == 2 and drift_matrix.shape[1] > 0:
        # Convert each column into a 1-D array as expected by PyKrige
        specified_drift = [np.asarray(drift_matrix)[:, i].ravel() for i in range(drift_matrix.shape[1])]
        uk_kwargs["drift_terms"] = ["specified"]
        uk_kwargs["specified_drift"] = specified_drift
        logger.info("Building UniversalKriging with %d specified drift term(s).", len(specified_drift))
    else:
        # For Ordinary Kriging, do NOT include 'drift_terms' in kwargs at all
        logger.info("Building OrdinaryKriging (no specified drift terms provided).")

    # Initialize the model
    try:
        uk_model = UniversalKriging(x, y, h, **uk_kwargs)
    except Exception as exc:
        logger.error("Failed to initialize UniversalKriging: %s", exc)
        raise

    return uk_model


def predict_at_points(uk_model, x, y, drift_matrix_pred=None):
    """
    Predict at points using a pykrige UniversalKriging model with strict drift validation.

    Parameters
    ----------
    uk_model : UniversalKriging-like
        Trained pykrige UniversalKriging instance (may be a test double).
    x, y : array-like
        1-D arrays of prediction coordinates.
    drift_matrix_pred : np.ndarray or None
        Drift matrix for prediction points (N_pred x n_terms) or None.

    Returns
    -------
    z, ss : np.ndarray, np.ndarray
        Predicted values and associated variances at provided points.

    Raises
    ------
    ValueError
        If the model was trained with specified drift but prediction drift is missing
        or column counts mismatch.
    """
    # Local import for defensive coding
    import numpy as _np

    x_arr = _np.asarray(x).ravel()
    y_arr = _np.asarray(y).ravel()

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length for prediction")

    # Determine if model was trained with specified drift
    try:
        # PyKrige stores drift_terms as a list of strings or None
        drift_terms = getattr(uk_model, "drift_terms", None)
        if isinstance(drift_terms, list) and "specified" in drift_terms:
            trained_with_drift = True
        else:
            # Fallback: check if specified_drift attribute is present and populated
            # Some PyKrige versions might not persist 'specified' in drift_terms correctly
            spec_drift = getattr(uk_model, "specified_drift", None)
            trained_with_drift = spec_drift is not None
    except Exception:
        trained_with_drift = False

    drift_pred = None
    if trained_with_drift:
        if drift_matrix_pred is None:
            raise ValueError("Model was trained with drift, but no drift provided for prediction.")
        drift_pred = _np.asarray(drift_matrix_pred)
        if drift_pred.ndim != 2:
            raise ValueError("drift_matrix_pred must be a 2-D array when model expects specified drift")

    # Best-effort: infer number of drift columns the model was trained with
    n_train_cols = None
    if trained_with_drift:
        for attr in ("specified_drift", "specified_drift_arrays", "specified_drift_data_arrays"):
            if hasattr(uk_model, attr):
                try:
                    cand = getattr(uk_model, attr)
                    n_train_cols = len(cand) if cand is not None else 0
                    break
                except Exception:
                    # ignore and try next
                    continue

    if trained_with_drift:
        n_pred_cols = int(drift_pred.shape[1])
        if n_train_cols is not None and n_pred_cols != n_train_cols:
            raise ValueError(
                f"Drift column count mismatch: model trained with {n_train_cols} columns but {n_pred_cols} provided"
            )

        # Build list of 1-D arrays as expected by pykrige
        specified_drift_arrays = [drift_pred[:, i].ravel() for i in range(drift_pred.shape[1])]

        try:
            # Try the modern/expected kwarg name
            result = uk_model.execute("points", x_arr, y_arr, specified_drift_arrays=specified_drift_arrays)
        except (TypeError, ValueError) as exc:
            # Fallback for other pykrige versions
            try:
                result = uk_model.execute("points", x_arr, y_arr, specified_drift=specified_drift_arrays)
            except Exception:
                # If both fail, re-raise the original exception for better debugging
                logger.error("uk_model.execute failed when passing specified drift: %s", exc)
                raise exc
        except Exception as exc:
            logger.error("uk_model.execute failed when passing specified drift: %s", exc)
            raise
    else:
        if drift_matrix_pred is not None and drift_matrix_pred.shape[1] > 0:
            logger.warning("Model was trained without drift; ignoring provided prediction drift.")
        try:
            result = uk_model.execute("points", x_arr, y_arr)
        except Exception as exc:
            logger.error("uk_model.execute failed: %s", exc)
            raise

    # Normalize return formats (pykrige has varied returns across versions)
    try:
        if isinstance(result, tuple) and len(result) >= 2:
            z = _np.asarray(result[0]).ravel()
            ss = _np.asarray(result[1]).ravel()
        elif isinstance(result, dict):
            # common keys: 'points', 'points_var' or 'z', 'ss'
            z = _np.asarray(result.get("points", result.get("z", []))).ravel()
            ss = _np.asarray(result.get("points_var", result.get("ss", []))).ravel()
        else:
            # attempt iterable unpack
            z, ss = result
            z = _np.asarray(z).ravel()
            ss = _np.asarray(ss).ravel()
    except Exception as exc:
        logger.error("Unexpected return from uk_model.execute: %s", exc)
        raise

    return z, ss


def predict_on_grid(uk_model, config: dict, term_names: list[str], resc: float, transform_params: dict | None = None, scaling_factors: dict | None = None):
    
    """
    
    Predict on a regular grid defined by `config['grid']` using the provided
    UniversalKriging model. This routine rebuilds the drift at each grid node
    with the exact `term_names` and `resc` value used during training, then
    calls `predict_at_points()` to obtain predictions and variances.
    Rebuilds drift using consistent scaling factors.
    
    Parameters
    ----------
    uk_model : UniversalKriging-like
        A trained pykrige UniversalKriging instance (or test double).
    config : dict
        Configuration dictionary containing a 'grid' mapping with keys
        'x_min','x_max','y_min','y_max','resolution'.
    term_names : list[str]
        Ordered list of drift term names that were used during training.
    resc : float
        Rescaling factor used to compute drift terms during training.
    transform_params : dict or None
        Parameters for coordinate transformation if anisotropy is enabled.

    Returns
    -------
    (GX, GY, Z_grid, SS_grid)
        Meshgrid arrays for X and Y and corresponding prediction and variance
        grids (same shape as the meshgrid).
        
    """
    
    import numpy as _np

    # Local imports to avoid hard dependencies at module import time
    try:
        from .drift import compute_drift_at_points
        from .transform import apply_transform
        from .AEM_drift import compute_linesink_drift_matrix
    except ImportError:
        from drift import compute_drift_at_points
        from transform import apply_transform
        from AEM_drift import compute_linesink_drift_matrix

    if not isinstance(config, dict):
        raise ValueError("config must be a dict containing a 'grid' entry")

    # 1. Grid Parameter Validation
    grid = config.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("config['grid'] must be a dict with x_min/x_max/y_min/y_max/resolution")

    try:
        x_min = float(grid["x_min"])
        x_max = float(grid["x_max"])
        y_min = float(grid["y_min"])
        y_max = float(grid["y_max"])
        res = float(grid["resolution"])
    except Exception as exc:
        logger.error("Invalid grid parameters: %s", exc)
        raise ValueError("Invalid or missing numeric grid parameters in config") from exc

    logger.info("Grid parameters: x_min=%.2f, x_max=%.2f, y_min=%.2f, y_max=%.2f, resolution=%.2f",
                x_min, x_max, y_min, y_max, res)

    if res <= 0.0:
        raise ValueError("grid resolution must be > 0")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("grid min must be less than grid max for both axes")

    # 2. Grid Generation
    # Build axes. Use np.arange with the documented endpoint behaviour.
    grid_x = _np.arange(x_min, x_max, res)
    grid_y = _np.arange(y_min, y_max, res)

    if grid_x.size == 0 or grid_y.size == 0:
        raise ValueError("Grid ranges and resolution produce empty axes")

    GX, GY = _np.meshgrid(grid_x, grid_y)
    flat_x = GX.ravel()
    flat_y = GY.ravel()

    # 3. Coordinate Transformation (necessary for Anisotropy)
    # Apply coordinate transformation if needed
    if transform_params is not None:
        logger.info("Applying coordinate transformation to prediction grid...")
        flat_x_model, flat_y_model = apply_transform(flat_x, flat_y, transform_params)
    else:
        flat_x_model, flat_y_model = flat_x, flat_y

    # 4. Unified Drift Matrix Reconstruction
    drift_columns = []
    poly_keywords = ['linear_x', 'linear_y', 'quadratic_x', 'quadratic_y']
    
    # Pre-load linesink data if linesink terms are present
    has_aem = any(t not in poly_keywords for t in term_names)
    linesink_data = None
    group_col = None
    sill = None
    
    # Configuration extraction for AEM
    apply_aniso = True
    strength_col = "resistance"
    rescaling_method = "adaptive"

    if has_aem:
        ls_source_cfg = config.get("data_sources", {}).get("linesink_river", {})
        import geopandas as gpd
        linesink_data = gpd.read_file(ls_source_cfg["path"])
        group_col = ls_source_cfg.get("group_column", "NAME")
        strength_col = ls_source_cfg.get("strength_col", "resistance")
        rescaling_method = ls_source_cfg.get("rescaling_method", "adaptive")

        # Parse Anisotropy Toggle from Config
        drift_term_conf = config.get("drift_terms", {}).get("linesink_river", False)
        if isinstance(drift_term_conf, dict):
            apply_aniso = drift_term_conf.get("apply_anisotropy", True)
        
        # Robust Sill Retrieval
        if hasattr(uk_model, 'variogram_parameters'):
            sill = float(uk_model.variogram_parameters.get('sill', 1.0))
        elif hasattr(uk_model, 'variogram_dict'):
            sill = float(uk_model.variogram_dict.get('sill', 1.0))
        else:
            sill = float(config.get("variogram", {}).get("sill", 1.0))

    # Slot each term into the matrix based on the master term_names list
    for term in term_names:
        if term in poly_keywords:
            # Slot: Polynomial Term
            col_data, _ = compute_drift_at_points(flat_x_model, flat_y_model, [term], resc)
            drift_columns.append(col_data)
        else:
            # Slot: AEM Linesink Term
            single_ls_gdf = linesink_data[linesink_data[group_col] == term]

            # LOGIC FIX: Select correct grid coords
            if apply_aniso:
                # River is transformed -> Use Transformed Grid
                grid_calc_x, grid_calc_y = flat_x_model, flat_y_model
            else:
                # River is Raw -> Use RAW Grid
                grid_calc_x, grid_calc_y = flat_x, flat_y

            # Compute potential with CONSISTENT factors
            col_data, _, _ = compute_linesink_drift_matrix(  # Expect 3 returns now
                grid_calc_x, grid_calc_y,
                single_ls_gdf,
                group_col,
                transform_params,
                sill,
                strength_col=strength_col,
                rescaling_method=rescaling_method,
                apply_anisotropy=apply_aniso,
                input_scaling_factors=scaling_factors # <--- Pass the factors here
            )
            drift_columns.append(col_data)

    # Combine columns
    if drift_columns:
        drift_grid = _np.hstack(drift_columns)
    else:
        drift_grid = _np.zeros((len(flat_x_model), 0))

    # 5. Prediction execution    
    z_flat, ss_flat = predict_at_points(uk_model, flat_x_model, flat_y_model, drift_grid)
    
    # Reshape back to grid
    Z_grid = _np.asarray(z_flat).reshape(GX.shape)
    SS_grid = _np.asarray(ss_flat).reshape(GX.shape)

    logger.info("Predicted on grid: %dx%d nodes", GX.shape[0], GX.shape[1])

    return GX, GY, Z_grid, SS_grid


def output_drift_coefficients(all_h, drift_matrix, term_names):
    """
    Compute OLS estimates of drift coefficients for diagnostic purposes.

    The function performs a least-squares fit of `drift_matrix` (design matrix)
    to observed values `all_h` and logs coefficients per term name. It is
    intentionally non-intrusive (returns coefficients for inspection but does
    not affect kriging state).

    Parameters
    ----------
    all_h : array-like
        Observed dependent variable values (length N matching drift_matrix rows).
    drift_matrix : np.ndarray or None
        Design matrix used for drift (shape N x p). If None or zero-width,
        the function returns None.
    term_names : list[str]
        Ordered list of term names corresponding to columns of the drift matrix.

    Returns
    -------
    np.ndarray | None
        Coefficient vector of length p (float dtype) if solvable, otherwise None.
    """
    import numpy as _np

    if drift_matrix is None:
        logger.info("No drift matrix provided; skipping OLS diagnostics.")
        return None

    dm = _np.asarray(drift_matrix)
    if dm.ndim != 2 or dm.shape[1] == 0:
        logger.info("Drift matrix has zero columns; nothing to estimate.")
        return None

    try:
        y = _np.asarray(all_h).ravel()
        if y.size != dm.shape[0]:
            raise ValueError("Length of all_h does not match number of rows in drift_matrix")

        coeffs, residuals, rank, s = _np.linalg.lstsq(dm, y, rcond=None)
        # Log coefficients with readable formatting
        for name, coef in zip(term_names or [], coeffs):
            try:
                logger.info("OLS coefficient for %-15s : %.6e", name, float(coef))
            except Exception:
                logger.info("OLS coefficient for %-15s : %s", name, coef)
        return coeffs
    except Exception as exc:
        logger.warning("Failed to compute OLS drift coefficients: %s", exc)
        return None


def cross_validate(all_x: np.ndarray, all_y: np.ndarray, all_h: np.ndarray, config: dict, variogram: Any) -> dict:
    """
    Leave-One-Out Cross Validation (LOOCV) for Universal Kriging.

    For each held-out point this routine:
    - recomputes resc using the training subset
    - recomputes polynomial drift for the training subset
    - retrains a UniversalKriging on the training subset
    - predicts at the held-out location using the appropriate drift

    Returns a dictionary with keys: rmse, mae, q1 (mean standardized error),
    q2 (variance of standardized errors), predictions, variances, observations.

    Parameters
    ----------
    all_x, all_y, all_h : np.ndarray
        Training coordinates and values (1-D arrays).
    config : dict
        Configuration dictionary containing drift_terms and other settings.
    variogram : object
        Variogram-like object with sill, range_, nugget attributes.

    Returns
    -------
    dict
        Dictionary with keys: rmse, mae, q1, q2, predictions, variances, observations.
    """
    import math

    from v2_Code.drift import compute_resc, compute_polynomial_drift, compute_drift_at_points

    x = np.asarray(all_x).ravel()
    y = np.asarray(all_y).ravel()
    h = np.asarray(all_h).ravel()

    if x.size != y.size or x.size != h.size:
        raise ValueError("all_x, all_y and all_h must have the same length")

    n = x.size
    if n < 3:
        logger.warning("Not enough points for LOOCV (need >=3). n=%d", n)
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "q1": float("nan"),
            "q2": float("nan"),
            "predictions": np.array([]),
            "variances": np.array([]),
            "observations": np.array([]),
        }

    preds = np.full(n, np.nan)
    vars_ = np.full(n, np.nan)

    # Obtain variogram numeric values defensively
    try:
        covmax = float(getattr(variogram, "sill", 0.0))
    except Exception:
        covmax = 0.0
    try:
        variogram_range = float(getattr(variogram, "range_", getattr(variogram, "range", 0.0)))
    except Exception:
        variogram_range = 0.0

    for i in range(n):
        # training mask
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        x_train = x[train_mask]
        y_train = y[train_mask]
        h_train = h[train_mask]

        # recompute resc on training subset
        try:
            resc = compute_resc(covmax, x_train, y_train, variogram_range)
        except Exception as exc:
            logger.error("compute_resc failed at LOOCV index %d: %s", i, exc)
            continue

        # recompute drift matrix for training
        try:
            drift_train, term_names = compute_polynomial_drift(x_train, y_train, config, resc)
        except Exception as exc:
            logger.error("compute_polynomial_drift failed at LOOCV index %d: %s", i, exc)
            continue

        # build and train UK model on training subset
        try:
            uk_loocv = build_uk_model(x_train, y_train, h_train, drift_train, variogram)
        except Exception as exc:
            logger.error("build_uk_model failed during LOOCV at index %d: %s", i, exc)
            continue

        # compute drift at the held-out point using the same term ordering and resc
        try:
            drift_val, _ = compute_drift_at_points(np.array([x[i]]), np.array([y[i]]), term_names, resc)
        except Exception as exc:
            logger.error("compute_drift_at_points failed for validation point %d: %s", i, exc)
            drift_val = None

        # perform prediction
        try:
            if drift_val is not None:
                z, ss = predict_at_points(uk_loocv, np.array([x[i]]), np.array([y[i]]), drift_val)
            else:
                z, ss = predict_at_points(uk_loocv, np.array([x[i]]), np.array([y[i]]))

            preds[i] = float(np.asarray(z).ravel()[0])
            vars_[i] = float(np.asarray(ss).ravel()[0])
        except Exception as exc:
            logger.error("Prediction failed during LOOCV at index %d: %s", i, exc)
            continue

        if (i + 1) % 10 == 0:
            logger.info("LOOCV progress: %d/%d", i + 1, n)

    # Compute error statistics using only successfully predicted indices
    valid_mask = ~np.isnan(preds)
    if not np.any(valid_mask):
        logger.warning("LOOCV produced no valid predictions")
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "q1": float("nan"),
            "q2": float("nan"),
            "predictions": preds,
            "variances": vars_,
            "observations": h,
        }

    obs = h[valid_mask]
    pred_vals = preds[valid_mask]
    var_vals = vars_[valid_mask]

    errors = pred_vals - obs
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # standardized errors: avoid division by zero by using a small floor
    std_errs = []
    for ev, vv in zip(errors, var_vals):
        try:
            if vv is None or (isinstance(vv, float) and (vv <= 1e-12 or math.isnan(vv))):
                std_errs.append(float('nan'))
            else:
                std_errs.append(float(ev) / float(np.sqrt(vv)))
        except Exception:
            std_errs.append(float('nan'))
    std_errs = np.asarray(std_errs)

    # compute Q1 (mean) and Q2 (variance) ignoring NaNs
    if np.all(np.isnan(std_errs)):
        q1 = float('nan')
        q2 = float('nan')
    else:
        q1 = float(np.nanmean(std_errs))
        q2 = float(np.nanvar(std_errs))

    logger.info(
        "LOOCV completed: RMSE=%.6f MAE=%.6f Q1=%.6f Q2=%.6f",
        rmse,
        mae,
        q1 if not math.isnan(q1) else float('nan'),
        q2 if not math.isnan(q2) else float('nan'),
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "q1": q1,
        "q2": q2,
        "predictions": preds,
        "variances": vars_,
        "observations": h,
    }


__all__ = ["build_uk_model", "predict_at_points", "predict_on_grid", "output_drift_coefficients", "cross_validate"]
