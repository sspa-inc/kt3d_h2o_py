# -*- coding: utf-8 -*-
"""
Task 4.4 -- EXAMPLE: Anisotropy Handling
=========================================
Demonstrates geometric anisotropy in Universal Kriging using the
pre-transformation approach.

Key insight demonstrated:
  In transform.py, angle_major specifies the direction of the MAJOR
  correlation axis (longer range) in azimuth convention (CW from North).
  The perpendicular direction has SHORTER correlation (minor axis).

  The transform stretches the PERPENDICULAR direction by 1/ratio, making
  distances appear larger there, which reduces effective correlation range.

  For angle_major=30 deg azimuth (N30E):
    - N30E direction is unchanged -> longer correlation (MAJOR axis)
    - Perpendicular (WNW/ESE) gets stretched -> shorter correlation (MINOR axis)

Scenario:
  - 20 synthetic observation points with TRUE anisotropic spatial correlation
    generated via a covariance matrix.
  - angle_major=30 deg (azimuth), ratio=0.3
  - Major correlation along N30E/S30W (along angle_major)
  - Minor correlation along WNW/ESE (perpendicular to angle_major)

Outputs:
  docs/examples/output/ex_anisotropy.svg   (multi-panel figure)

Run from the project root:
    python docs/examples/ex_anisotropy.py
"""

import sys
import os
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- path setup --------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transform import get_transform_params, apply_transform, invert_transform_coords
from kriging import build_uk_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("Task 4.4 -- EXAMPLE: Anisotropy Handling")
print("=" * 65)


# ---- Variogram stub ----------------------------------------------------------
class SimpleVariogram:
    """Minimal variogram object compatible with build_uk_model()."""
    def __init__(self, model="spherical", sill=1.0, range_=100.0, nugget=0.1,
                 anisotropy_enabled=False, anisotropy_ratio=1.0, angle_major=0.0):
        self.model_type = model
        self.sill = sill
        self.range_ = range_
        self.nugget = nugget
        self.anisotropy_enabled = anisotropy_enabled
        self.anisotropy_ratio = anisotropy_ratio
        self.angle_major = angle_major

    @property
    def model(self):
        return self.model_type

    def clone(self):
        import copy
        return copy.deepcopy(self)


# =============================================================================
# 1. GENERATE SYNTHETIC DATA WITH TRUE ANISOTROPIC CORRELATION
#
# The pre-transformation in transform.py works as follows:
#   1. Center at centroid
#   2. Convert azimuth to arithmetic: theta = 90 - angle_deg
#   3. Rotate using coords @ R: angle_major direction maps to X-axis
#   4. Scale Y by 1/ratio: Y-axis distances get STRETCHED
#
# Therefore:
#   - Direction at angle_major -> X -> unchanged -> LONGER effective range (MAJOR)
#   - Direction perpendicular   -> Y -> stretched -> SHORTER effective range (MINOR)
#
# For angle_major=30 azimuth (N30E):
#   - N30E direction: effective range = range_major                (major axis)
#   - WNW/ESE direction: effective range = range_major * ratio     (minor axis)
#
# We generate synthetic data using the SAME transform to compute effective
# distances, ensuring the covariance structure matches what kriging expects.
# =============================================================================
print("\n[1] Generating synthetic data with anisotropic spatial correlation...")

rng = np.random.default_rng(42)
N = 20

# Anisotropy parameters
ANGLE_MAJOR = 30.0   # degrees CW from North (azimuth) -- 30 deg azimuth = N30E direction
RATIO       = 0.3    # minor_range / major_range
SILL        = 1.0
RANGE_MAJOR = 120.0  # effective range along angle_major direction (major axis)
NUGGET      = 0.05

# Convert azimuth to arithmetic for all plotting/trig operations
ANGLE_ARITH = 90.0 - ANGLE_MAJOR  # = 60 deg arithmetic (CCW from East)

# Derived ranges in raw space
RANGE_MINOR = RANGE_MAJOR * RATIO   # effective range perpendicular to angle_major
# Minor correlation direction is perpendicular to angle_major
MINOR_DIR_ARITH = ANGLE_ARITH + 90.0  # = 150 deg arithmetic (WNW/ESE)
MINOR_DIR_AZ = ANGLE_MAJOR + 90.0     # = 120 deg azimuth

x_raw = rng.uniform(10, 190, N)
y_raw = rng.uniform(10, 190, N)

# Use the SAME transform as transform.py to compute model-space distances
# get_transform_params handles azimuth->arithmetic conversion internally
params = get_transform_params(x_raw, y_raw, angle_deg=ANGLE_MAJOR, ratio=RATIO)
x_model, y_model = apply_transform(x_raw, y_raw, params)

# Pairwise distances in model (isotropic) space
coords_model = np.column_stack([x_model, y_model])
D = cdist(coords_model, coords_model)

# Spherical covariance
psill = SILL - NUGGET
h_norm = D / RANGE_MAJOR
cov_matrix = np.where(
    h_norm <= 1.0,
    SILL - NUGGET - psill * (1.5 * h_norm - 0.5 * h_norm**3),
    0.0
)
np.fill_diagonal(cov_matrix, SILL)
cov_matrix += np.eye(N) * 1e-8

# Generate correlated random field
L = np.linalg.cholesky(cov_matrix)
z_true = L @ rng.standard_normal(N) + 10.0

# Precompute trig for drawing (arithmetic convention)
theta_arith_rad = np.radians(ANGLE_ARITH)
cos_maj, sin_maj = np.cos(theta_arith_rad), np.sin(theta_arith_rad)  # major axis direction
minor_dir_rad = np.radians(MINOR_DIR_ARITH)
cos_min, sin_min = np.cos(minor_dir_rad), np.sin(minor_dir_rad)  # minor axis direction

print("   Points: N=%d, domain=[10,190]x[10,190]" % N)
print("   angle_major=%.0f deg azimuth (N30E) -- MAJOR correlation axis (unchanged by transform)" % ANGLE_MAJOR)
print("   Minor correlation axis: %.0f deg azimuth (WNW/ESE) -- perpendicular, gets STRETCHED" % MINOR_DIR_AZ)
print("   ratio=%.1f, range_major=%.0f (along N30E), range_minor=%.0f (along WNW/ESE)" % (
    RATIO, RANGE_MAJOR, RANGE_MINOR))
print("   z range: [%.3f, %.3f]" % (z_true.min(), z_true.max()))

# =============================================================================
# 2. VERIFY TRANSFORM BEHAVIOR
# =============================================================================
print("\n[2] Verifying transform behavior...")

# Direction along angle_major (30 deg azimuth = 60 deg arithmetic)
# Unit vector: (cos(60), sin(60)) = (0.5, 0.866) -- should be UNCHANGED
along_dx, along_dy = cos_maj, sin_maj
along_x, along_y = apply_transform(
    np.array([along_dx]) + params['center'][0],
    np.array([along_dy]) + params['center'][1], params)
along_dist = np.sqrt(along_x[0]**2 + along_y[0]**2)

# Direction perpendicular to angle_major (120 deg azimuth = 150 deg arithmetic)
# Should be STRETCHED by 1/ratio
perp_dx, perp_dy = cos_min, sin_min
perp_x, perp_y = apply_transform(
    np.array([perp_dx]) + params['center'][0],
    np.array([perp_dy]) + params['center'][1], params)
perp_dist = np.sqrt(perp_x[0]**2 + perp_y[0]**2)

print("   Along angle_major (%.0f\u00b0 az) -> model dist=%.3f (unchanged, major axis)" % (
    ANGLE_MAJOR, along_dist))
print("   Perpendicular (%.0f\u00b0 az) -> model dist=%.3f (stretched by 1/ratio=%.1f, minor axis)" % (
    MINOR_DIR_AZ, perp_dist, 1.0/RATIO))
print("   Ratio of distances (perp/along): %.2f (should be ~%.2f = 1/ratio)" % (
    perp_dist/along_dist, 1.0/RATIO))

x_back, y_back = invert_transform_coords(x_model, y_model, params)
roundtrip_err = np.max(np.abs(x_back - x_raw) + np.abs(y_back - y_raw))
print("   Roundtrip error: %.2e" % roundtrip_err)

# =============================================================================
# 3. BUILD VARIOGRAM OBJECTS
# =============================================================================
print("\n[3] Building variogram objects...")

# Clone with anisotropy disabled -- passed to PyKrige after pre-transform
vario_iso_clone = SimpleVariogram(
    model="spherical", sill=SILL, range_=RANGE_MAJOR, nugget=NUGGET,
    anisotropy_enabled=False
)

vario_no_aniso = SimpleVariogram(
    model="spherical", sill=SILL, range_=RANGE_MAJOR, nugget=NUGGET,
    anisotropy_enabled=False
)

print("   vario_iso_clone: for aniso run (pre-transformed coords, isotropic kriging)")
print("   vario_no_aniso:  for isotropic comparison (raw coords)")

# =============================================================================
# 4. BUILD MODELS AND PREDICT ON GRID
# =============================================================================
print("\n[4] Building kriging models and predicting on grid...")

grid_res = 3.0
gx_1d = np.arange(10, 190, grid_res)
gy_1d = np.arange(10, 190, grid_res)
GX, GY = np.meshgrid(gx_1d, gy_1d)
flat_x = GX.ravel()
flat_y = GY.ravel()

# ---- Run A: Anisotropy pre-transformation ------------------------------------
print("   Run A: With anisotropy (pre-transform)...")
uk_aniso = build_uk_model(x_model, y_model, z_true,
                          drift_matrix=None, variogram=vario_iso_clone)
flat_x_m, flat_y_m = apply_transform(flat_x, flat_y, params)
z_aniso_flat, ss_aniso_flat = uk_aniso.execute("points", flat_x_m, flat_y_m)
Z_aniso = z_aniso_flat.reshape(GX.shape)
SS_aniso = ss_aniso_flat.reshape(GX.shape)
print("   Run A: Z range=[%.3f, %.3f]" % (Z_aniso.min(), Z_aniso.max()))

# ---- Run B: No anisotropy (isotropic) ----------------------------------------
print("   Run B: Without anisotropy (isotropic)...")
uk_iso = build_uk_model(x_raw, y_raw, z_true,
                        drift_matrix=None, variogram=vario_no_aniso)
z_iso_flat, ss_iso_flat = uk_iso.execute("points", flat_x, flat_y)
Z_iso = z_iso_flat.reshape(GX.shape)
SS_iso = ss_iso_flat.reshape(GX.shape)
print("   Run B: Z range=[%.3f, %.3f]" % (Z_iso.min(), Z_iso.max()))

# =============================================================================
# 5. MULTI-PANEL FIGURE (2 rows x 3 cols)
# =============================================================================
print("\n[5] Generating figure...")

fig = plt.figure(figsize=(17, 11))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.35, wspace=0.30,
                       left=0.05, right=0.97, top=0.89, bottom=0.06)

CMAP_PRED = "RdYlBu_r"
CMAP_VAR = "YlOrRd"

vmin_z = min(Z_aniso.min(), Z_iso.min())
vmax_z = max(Z_aniso.max(), Z_iso.max())

# ---- Panel (0,0): Raw point cloud with correlation ellipse -------------------
ax0 = fig.add_subplot(gs[0, 0])
sc0 = ax0.scatter(x_raw, y_raw, c=z_true, cmap=CMAP_PRED, s=60,
                  edgecolors="k", lw=0.5, zorder=3)
plt.colorbar(sc0, ax=ax0, label="z", shrink=0.85)

# Draw correlation ellipse.
# Major axis along angle_major direction: ANGLE_ARITH = 60 deg arithmetic (N30E)
# Minor axis perpendicular: MINOR_DIR_ARITH = 150 deg arithmetic (WNW/ESE)
cx, cy = params['center']
n_ell = 200
ell_t = np.linspace(0, 2 * np.pi, n_ell)

# Parametric ellipse: P(t) = center + a*cos(t)*u_major + b*sin(t)*u_minor
u_major = np.array([cos_maj, sin_maj])  # unit vector along major axis (N30E)
u_minor = np.array([cos_min, sin_min])  # unit vector along minor axis (WNW/ESE)
ell_pts = (cx + RANGE_MAJOR * np.outer(np.cos(ell_t), u_major)
           + RANGE_MINOR * np.outer(np.sin(ell_t), u_minor))
ax0.plot(ell_pts[:, 0], ell_pts[:, 1], 'b-', lw=1.5, alpha=0.7,
         label='Correlation ellipse')

# Major axis line (N30E direction, 60 deg arithmetic = angle_major)
ax_half = 80
ax0.plot([cx - ax_half * cos_maj, cx + ax_half * cos_maj],
         [cy - ax_half * sin_maj, cy + ax_half * sin_maj],
         'b-', lw=2.0, label='Major axis (N30E, range=%d)' % int(RANGE_MAJOR))

# Minor axis line (WNW/ESE direction, 150 deg arithmetic = perpendicular)
mi_half = 80 * RATIO
ax0.plot([cx - mi_half * cos_min, cx + mi_half * cos_min],
         [cy - mi_half * sin_min, cy + mi_half * sin_min],
         'r--', lw=1.5, label='Minor axis (WNW/ESE, range=%d)' % int(RANGE_MINOR))

# Angle arc from North (90 deg arithmetic) CW to angle_major direction (60 deg arithmetic)
# In arithmetic coords, CW from North means going from 90 deg down to 60 deg
arc_r = 30
arc_a = np.linspace(np.radians(ANGLE_ARITH), np.radians(90.0), 40)
ax0.plot(cx + arc_r * np.cos(arc_a), cy + arc_r * np.sin(arc_a), 'r-', lw=1.2)
# Label at midpoint of arc
mid_arc = np.radians((ANGLE_ARITH + 90.0) / 2.0)
ax0.text(cx + arc_r * 1.3 * np.cos(mid_arc),
         cy + arc_r * 1.3 * np.sin(mid_arc),
         "angle_major\n=30\u00b0 az", fontsize=7, color="red", ha="center")
# North reference arrow
ax0.annotate("", xy=(cx, cy + 40), xytext=(cx, cy),
             arrowprops=dict(arrowstyle="-|>", color="gray", lw=1))
ax0.text(cx + 3, cy + 42, "N (0\u00b0 az)", color="gray", fontsize=7, va="bottom")

ax0.set_xlim(-5, 210)
ax0.set_ylim(-5, 210)
ax0.set_xlabel("X (raw space)")
ax0.set_ylabel("Y (raw space)")
ax0.set_title("Raw Point Cloud with Correlation Ellipse", fontsize=10, fontweight="bold")
ax0.set_aspect("equal")
ax0.legend(fontsize=6.5, loc="lower right")
ax0.grid(True, alpha=0.2)

# ---- Panel (0,1): Explanation text -------------------------------------------
ax1 = fig.add_subplot(gs[0, 1])
ax1.axis("off")

explanation = (
    "ANGLE CONVENTION (Azimuth, CW from North)\n"
    "  0 deg = North (+Y)   90 deg = East (+X)\n"
    "  30 deg = N30E         120 deg = ESE\n"
    "\n"
    "ANISOTROPY PARAMETERS\n"
    "  angle_major = 30 deg (azimuth, CW from North)\n"
    "  ratio = 0.3\n"
    "\n"
    "HOW THE TRANSFORM WORKS\n"
    "  1. Center at centroid\n"
    "  2. Convert azimuth to arithmetic: theta = 90 - 30 = 60 deg\n"
    "  3. Rotate so angle_major direction maps to X-axis\n"
    "  4. Scale Y by 1/ratio = 3.3\n"
    "     -> Y (perpendicular) distances STRETCHED\n"
    "\n"
    "EFFECT ON CORRELATION\n"
    "  angle_major direction (N30E, 30 deg az):\n"
    "    -> mapped to X -> unchanged -> LONGER range\n"
    "    -> effective range = %d (MAJOR axis)\n"
    "  Perpendicular direction (WNW/ESE, 120 deg az):\n"
    "    -> mapped to Y -> stretched -> SHORTER range\n"
    "    -> effective range = %d * 0.3 = %d (MINOR axis)\n"
    "\n"
    "WHY CLONE WITH anisotropy_enabled=False?\n"
    "  After pre-transforming, PyKrige must NOT\n"
    "  apply its own anisotropy (double-transform).\n"
    "  Clone variogram with enabled=False."
) % (int(RANGE_MAJOR), int(RANGE_MAJOR), int(RANGE_MINOR))

ax1.text(0.02, 0.98, explanation, transform=ax1.transAxes,
         fontsize=7.5, va="top", ha="left", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef6fb",
                   edgecolor="#4a90d9", lw=1.2))
ax1.set_title("Explanation", fontsize=10, fontweight="bold")

# ---- Panel (0,2): Difference map (aniso - iso) ------------------------------
ax2 = fig.add_subplot(gs[0, 2])
diff = Z_aniso - Z_iso
abs_max = max(np.abs(diff).max(), 1e-6)
im2 = ax2.pcolormesh(GX, GY, diff, cmap="RdBu_r",
                     vmin=-abs_max, vmax=abs_max, shading="auto")
plt.colorbar(im2, ax=ax2, label="dz (aniso - iso)", shrink=0.85)
ax2.scatter(x_raw, y_raw, c="k", s=20, edgecolors="white", lw=0.4, zorder=3)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Difference: Aniso - Iso\n(impact of anisotropy assumption)",
              fontsize=10, fontweight="bold")
ax2.set_aspect("equal")

# ---- Panel (1,0): Prediction WITH anisotropy ---------------------------------
ax3 = fig.add_subplot(gs[1, 0])
im3 = ax3.pcolormesh(GX, GY, Z_aniso, cmap=CMAP_PRED,
                     vmin=vmin_z, vmax=vmax_z, shading="auto")
plt.colorbar(im3, ax=ax3, label="z", shrink=0.85)
ax3.scatter(x_raw, y_raw, c="white", s=20, edgecolors="k", lw=0.4, zorder=3)
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_title("Prediction WITH Anisotropy\n(major correlation along N30E)",
              fontsize=10, fontweight="bold")
ax3.set_aspect("equal")

# ---- Panel (1,1): Prediction WITHOUT anisotropy ------------------------------
ax4 = fig.add_subplot(gs[1, 1])
im4 = ax4.pcolormesh(GX, GY, Z_iso, cmap=CMAP_PRED,
                     vmin=vmin_z, vmax=vmax_z, shading="auto")
plt.colorbar(im4, ax=ax4, label="z", shrink=0.85)
ax4.scatter(x_raw, y_raw, c="white", s=20, edgecolors="k", lw=0.4, zorder=3)
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.set_title("Prediction WITHOUT Anisotropy\n(isotropic, same variogram range)",
              fontsize=10, fontweight="bold")
ax4.set_aspect("equal")

# ---- Panel (1,2): Variance comparison ----------------------------------------
ax5 = fig.add_subplot(gs[1, 2])
vmax_ss = max(SS_aniso.max(), SS_iso.max())
im5 = ax5.pcolormesh(GX, GY, SS_aniso, cmap=CMAP_VAR,
                     vmin=0, vmax=vmax_ss, shading="auto")
plt.colorbar(im5, ax=ax5, label="variance (aniso)", shrink=0.85)
cs = ax5.contour(GX, GY, SS_iso, levels=5, colors='blue', linewidths=0.8,
                 linestyles='dashed')
ax5.clabel(cs, fontsize=6, fmt="%.2f")
ax5.scatter(x_raw, y_raw, c="white", s=20, edgecolors="k", lw=0.4, zorder=3)
ax5.set_xlabel("X")
ax5.set_ylabel("Y")
ax5.set_title("Variance: Aniso (color) vs Iso (blue contours)\n"
              "(aniso zones elongated along N30E)",
              fontsize=10, fontweight="bold")
ax5.set_aspect("equal")

# ---- Main title --------------------------------------------------------------
fig.suptitle(
    "Task 4.4 -- Anisotropy Handling Example\n"
    "angle_major=30\u00b0 azimuth (N30E)  |  ratio=0.3  |  "
    "Major correlation along N30E (angle_major direction)",
    fontsize=11, fontweight="bold", y=0.96
)

out_path = os.path.join(OUTPUT_DIR, "ex_anisotropy.svg")
fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=150)
plt.close(fig)
print("\n   Figure saved: %s" % out_path)

# =============================================================================
# 6. SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("  angle_major=30 deg CW from North (azimuth) -- N30E direction")
print("    -> This direction is UNCHANGED in model space (MAJOR axis)")
print("    -> Effective range along N30E = %d (MAJOR axis)" % int(RANGE_MAJOR))
print("  Perpendicular direction = 120 deg azimuth (WNW/ESE)")
print("    -> STRETCHED by 1/ratio in model space (MINOR axis)")
print("    -> Effective range along WNW/ESE = %d (MINOR axis)" % int(RANGE_MINOR))
print("")
print("  Pre-transformation: translate -> rotate (az 30 -> arith 60) -> scale Y by %.1f" % (1.0/RATIO))
print("  angle_major direction -> X-axis (unchanged)")
print("  Perpendicular direction -> Y-axis (stretched)")
print("  Variogram clone: anisotropy_enabled=False (prevents double-transform)")
print("  Roundtrip error: %.2e" % roundtrip_err)
print("  Output: %s" % out_path)
print("=" * 65)
