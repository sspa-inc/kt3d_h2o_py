import numpy as np
import logging
from transform import apply_transform

logger = logging.getLogger(__name__)

def compute_linesink_potential(x, y, x1, y1, x2, y2, strength=1.0):
    """
    Implements the potential for a line sink analytic element based on LS_aem_sub. [cite: 118]
    Calculates the potential at (x, y) coordinates. [cite: 125]
    """
    # Calculate length of the line segment [cite: 125, 151]
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if L < 1e-6:
        return np.zeros_like(x)

    # Use complex numbers for analytic element potential [cite: 125, 126]
    z = x + 1j * y
    z1 = x1 + 1j * y1
    z2 = x2 + 1j * y2
    
    # Midpoint and half-length vector to map to ZZ space [-1, 1] [cite: 126]
    mid = (z1 + z2) / 2.0
    half_L_vec = (z2 - z1) / 2.0
    ZZ = (z - mid) / half_L_vec
    
    # Trap singularities at endpoints where ZZ is near +/- 1.0 [cite: 127]
    small = 1e-10
    mask = np.abs(ZZ - 1.0) < small
    ZZ[mask] += small

    # Protect Z = -1 (Start of segment) <-- MISSING IN YOUR VERSION
    mask_neg = np.abs(ZZ + 1.0) < small
    ZZ[mask_neg] = -1.0 - small

    # Compute complex potential (carg) [cite: 127]
    # Function: (ZZ+1)log(ZZ+1) - (ZZ-1)log(ZZ-1) + 2log((z2-z1)/2) - 2
    term1 = (ZZ + 1.0) * np.log(ZZ + 1.0)
    term2 = (ZZ - 1.0) * np.log(ZZ - 1.0)
    term3 = 2.0 * np.log(half_L_vec)
    carg = term1 - term2 + term3 - 2.0
    
    # Return real part (potential) [cite: 127]
    # Strength is multiplied by Length and divided by 4*PI [cite: 127]
    phi = (strength * L / (4.0 * np.pi)) * np.real(carg)
    return phi

#def compute_linesink_drift_matrix(x_model, y_model, linesinks_gdf, group_col, transform_params, sill, strength_col='lval'):
#def compute_linesink_drift_matrix(x_model, y_model, linesinks_gdf, group_col, transform_params, sill, strength_col='lval', rescaling_method='adaptive'):
# def compute_linesink_drift_matrix(x_model, y_model, linesinks_gdf, group_col, 
#                                   transform_params, sill, strength_col='lval', 
#                                   rescaling_method='adaptive', apply_anisotropy=True):
def compute_linesink_drift_matrix(x_model, y_model, linesinks_gdf, group_col, 
                                  transform_params, sill, strength_col='lval', 
                                  rescaling_method='adaptive', apply_anisotropy=True,
                                  input_scaling_factors=None): # <--- 1. NEW ARGUMENT

    """
    
    Groups linesink segments by a string attribute and transforms them to model space.
    Computes drift matrix with consistent scaling support.    
    
    Parameters:
    -----------
    x_model, y_model : np.ndarray
        Coordinates of points in model space.
    linesinks_gdf : GeoDataFrame
        Shapefile data containing linesink geometries and IDs.
    group_col : str
        The string attribute (e.g., 'NAME') used to group linesink segments.
    transform_params : dict
        Parameters from transform.py to align segments with anisotropy. [cite: 17]
    sill : float
        Variogram sill used for AEM-specific rescaling. [cite: 535]
    strength_col : str, optional
        `Column name in linesinks_gdf containing the strength values for each linesink segment, by default 'lval'.
    rescaling_method : str
        'adaptive' (matches UK_SSPA, safer) or 'fixed' (matches KT3D/current AEM_drift, aggressive).        
    apply_anisotropy : bool
        If True, applies transform_params to line sink geometry (AEM style).
        If False, uses raw coordinates (KT3D style).


    """
    
    # 1. Handle the toggle:
    # If anisotropy is disabled for this specific term, we treat the transform as None.
    # transform.py's apply_transform returns raw coords if params is None.
    current_transform = transform_params if apply_anisotropy else None
    
    unique_ids = linesinks_gdf[group_col].unique()
    drift_matrix = np.zeros((len(x_model), len(unique_ids)))    
    
    scaling_factors_used = {}

    # AEM specific rescaling logic from KT3D [cite: 535]
    # rescr = dble(covmax/(0.0001000D0))
    # rescr = sill / 0.0001

    for i, linesink_id in enumerate(unique_ids):
        segments = linesinks_gdf[linesinks_gdf[group_col] == linesink_id]
        total_phi = np.zeros_like(x_model)
        
        for _, row in segments.iterrows():
            coords = list(row.geometry.coords)
            for j in range(len(coords) - 1):

                p1_raw = np.array([coords[j][0]]), np.array([coords[j][1]])
                p2_raw = np.array([coords[j+1][0]]), np.array([coords[j+1][1]])
                
                # 2. Use the local variable 'current_transform'
                x1_m, y1_m = apply_transform(p1_raw[0], p1_raw[1], current_transform)
                x2_m, y2_m = apply_transform(p2_raw[0], p2_raw[1], current_transform)
                
                strength = row.get(strength_col, 1.0)

                # print(x_model, y_model, x1_m, y1_m, x2_m, y2_m, strength)
                
                total_phi += compute_linesink_potential(
                    x_model, y_model, x1_m[0], y1_m[0], x2_m[0], y2_m[0], strength
                )

        # --- NEW SCALING LOGIC ---
        # 1. If factors are provided (Prediction Phase), use them.
        if input_scaling_factors and linesink_id in input_scaling_factors:
            rescr = input_scaling_factors[linesink_id]
            
        # 2. If 'fixed' is requested (KT3D style), use constant.
        elif rescaling_method == 'fixed':
            # Reduced from 0.0001 to 1.0 to prevent matrix explosion in metric units
            rescr = sill / 0.0001 
            
        # 3. Default 'adaptive' (Training Phase): Calculate and save.
        else:
            max_abs = np.max(np.abs(total_phi))
            if max_abs > 1e-10:
                rescr = sill / max_abs
            else:
                rescr = 1.0

        # Store factor for return
        scaling_factors_used[linesink_id] = rescr
        
        drift_matrix[:, i] = total_phi * rescr
        logger.info(f"Generated drift column for '{linesink_id}'. Factor: {rescr:.4e}")

    # <--- 2. RETURN FACTORS
    return drift_matrix, list(unique_ids), scaling_factors_used