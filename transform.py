"""
Coordinate transformation utilities for UK_SSPA v2.

Handles affine transformations (translation, rotation, scaling) to align
geometric anisotropy with drift term calculation.
"""
from __future__ import annotations
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

def get_transform_params(x: np.ndarray, y: np.ndarray, angle_deg: float, ratio: float) -> dict:
    """
    Compute transformation parameters for anisotropy.

    Parameters
    ----------
    x, y : np.ndarray
        Original coordinates (used to compute centroid).
    angle_deg : float
        Rotation angle in degrees (Azimuth: Clockwise from North, 0°=North).
    ratio : float
        Anisotropy ratio (minor_range / major_range), 0 < ratio <= 1.

    Returns
    -------
    dict
        Dictionary containing 'center', 'R' (rotation matrix), and 'S' (scaling vector).
    """
    # 1. Centroid (Translation)
    x_center = np.mean(x)
    y_center = np.mean(y)

    # 2. Rotation Matrix
    # Convert azimuth (CW from North) to arithmetic (CCW from East), then to radians
    theta = math.radians(90.0 - angle_deg)

    # Standard 2D rotation matrix (Counter-Clockwise)
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([
        [c, -s],
        [s,  c]
    ])

    # 3. Scaling Matrix
    # We stretch the minor axis to match the major axis.
    # Strategy: Rotate data so Major Axis aligns with X-axis.
    # Then scale Y-axis (minor) by 1/ratio to make it isotropic.
    inv_ratio = 1.0 / ratio if ratio > 0 else 1.0
    S = np.array([1.0, inv_ratio])

    logger.info(f"Transform Params: Center=({x_center:.2f}, {y_center:.2f}), "
                f"Angle={angle_deg:.1f}, Ratio={ratio:.2f}")

    return {
        "center": np.array([x_center, y_center]),
        "R": R,
        "S": S
    }

def apply_transform(x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply affine transformation to coordinates.

    Formula: X_prime = S * (R * (X - Center))
    """
    if params is None:
        return x, y

    # Stack into (N, 2) matrix
    coords = np.column_stack((x, y))

    # 1. Translate
    coords_centered = coords - params["center"]

    # 2. Rotate
    coords_rotated = np.dot(coords_centered, params["R"])

    # 3. Scale
    coords_transformed = coords_rotated * params["S"]

    return coords_transformed[:, 0], coords_transformed[:, 1]

def invert_transform_coords(x_prime: np.ndarray, y_prime: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform from model space back to real-world coordinates.
    """
    if params is None:
        return x_prime, y_prime

    coords_prime = np.column_stack((x_prime, y_prime))

    # 1. Un-Scale
    coords_unscaled = coords_prime / params["S"]

    # 2. Un-Rotate
    coords_unrotated = np.dot(coords_unscaled, params["R"].T)

    # 3. Un-Translate
    coords_original = coords_unrotated + params["center"]

    return coords_original[:, 0], coords_original[:, 1]
