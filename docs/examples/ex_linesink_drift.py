"""
Task 4.3 — EXAMPLE: Incorporating River Effects with AEM Linesink Drift
=======================================================================
Demonstrates the AEM linesink drift feature for Universal Kriging.

Scenario:
  - A synthetic river with 3 line segments is created as a shapefile
    (written to a temp directory so no external files are needed).
  - 25 observation points with water levels influenced by proximity to river.
  - AEM linesink drift is enabled with apply_anisotropy=True.
  - Shows the raw AEM potential field before and after scaling.
  - Shows the combined drift matrix (polynomial + AEM).
  - Explains trained_scaling_factors and why they must persist to prediction.

Outputs:
  docs/examples/output/ex_linesink_drift.svg   (multi-panel figure)

Run from the project root:
    python docs/examples/ex_linesink_drift.py
"""

import sys
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
from shapely.geometry import LineString

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AEM_drift import compute_linesink_drift_matrix, compute_linesink_potential
from drift import compute_resc, compute_polynomial_drift
from kriging import build_uk_model
from transform import get_transform_params, apply_transform

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Task 4.3 — EXAMPLE: AEM Linesink Drift")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Create a synthetic river shapefile (3 segments, 2 groups)
# ─────────────────────────────────────────────────────────────────────────────
# The river runs roughly east-west across the domain [0,100]x[0,100].
# Group "RiverA" = main channel (2 segments), Group "RiverB" = tributary (1 segment).

print("\n[1] Creating synthetic river shapefile...")

river_segments = [
    # Group "RiverA" — main channel, two connected segments
    {"geometry": LineString([(10, 60), (40, 55)]), "group": "RiverA", "resistance": 1.0},
    {"geometry": LineString([(40, 55), (80, 50)]), "group": "RiverA", "resistance": 1.0},
    # Group "RiverB" — tributary, one segment
    {"geometry": LineString([(30, 80), (45, 60)]), "group": "RiverB", "resistance": 0.6},
]

river_gdf = gpd.GeoDataFrame(river_segments, crs="EPSG:32632")

# Write to a temporary shapefile so the example is fully self-contained
_tmpdir = tempfile.mkdtemp()
river_shp_path = os.path.join(_tmpdir, "synthetic_river.shp")
river_gdf.to_file(river_shp_path)
print(f"   River shapefile written to: {river_shp_path}")
print(f"   Groups: {river_gdf['group'].unique().tolist()}")
print(f"   Segments: {len(river_gdf)}")

# Reload from disk (mirrors real pipeline usage)
river_gdf = gpd.read_file(river_shp_path)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Synthetic observation points — water levels influenced by river
# ─────────────────────────────────────────────────────────────────────────────
# True field: water level declines away from the river.
# We approximate river influence using the AEM potential of RiverA.

print("\n[2] Generating 25 synthetic observation points...")

rng = np.random.default_rng(42)
N = 25
x_obs = rng.uniform(5, 95, N)
y_obs = rng.uniform(5, 95, N)

# Compute "true" AEM potential at observation points (RiverA only, for truth)
phi_true_A = np.zeros(N)
for _, row in river_gdf[river_gdf["group"] == "RiverA"].iterrows():
    coords = list(row.geometry.coords)
    for j in range(len(coords) - 1):
        phi_true_A += compute_linesink_potential(
            x_obs, y_obs,
            coords[j][0], coords[j][1],
            coords[j+1][0], coords[j+1][1],
            strength=row["resistance"]
        )

# Normalise to [0, 5] range and add noise
phi_norm = (phi_true_A - phi_true_A.min()) / (phi_true_A.max() - phi_true_A.min() + 1e-12)
z_obs = 10.0 + 5.0 * phi_norm + rng.normal(0, 0.3, N)

print(f"   x range: [{x_obs.min():.1f}, {x_obs.max():.1f}]")
print(f"   y range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
print(f"   z range: [{z_obs.min():.2f}, {z_obs.max():.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Variogram and transform parameters
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVariogram:
    """Minimal variogram object compatible with build_uk_model()."""
    def __init__(self, model="spherical", sill=1.0, range_=50.0, nugget=0.1):
        self.model_type = model
        self.sill = sill
        self.range_ = range_
        self.nugget = nugget
        self.anisotropy_enabled = False
        self.anisotropy_ratio = 1.0
        self.angle_major = 0.0

    @property
    def model(self):
        return self.model_type


SILL   = 2.0
RANGE  = 40.0
NUGGET = 0.1

vario = SimpleVariogram(model="spherical", sill=SILL, range_=RANGE, nugget=NUGGET)

# Anisotropy: mild, angle=30° CCW from East, ratio=0.7
ANGLE_DEG = 30.0
RATIO     = 0.7
transform_params = get_transform_params(x_obs, y_obs, ANGLE_DEG, RATIO)

# Transform observation coordinates to model space
x_model, y_model = apply_transform(x_obs, y_obs, transform_params)

print(f"\n[3] Variogram: spherical  sill={SILL}  range={RANGE}  nugget={NUGGET}")
print(f"    Anisotropy: angle={ANGLE_DEG}° CCW from East, ratio={RATIO}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Compute AEM linesink drift matrix (TRAINING)
# ─────────────────────────────────────────────────────────────────────────────
# apply_anisotropy=True → linesink geometry is also transformed to model space.
# rescaling_method='adaptive' → each group's potential is normalised to the sill.

print("\n[4] Computing AEM drift matrix at training points...")

aem_matrix_train, aem_names, trained_scaling_factors = compute_linesink_drift_matrix(
    x_model, y_model,
    river_gdf,
    group_col="group",
    transform_params=transform_params,
    sill=SILL,
    strength_col="resistance",
    rescaling_method="adaptive",
    apply_anisotropy=True,
    input_scaling_factors=None   # None → training phase, factors are computed
)

print(f"   AEM drift matrix shape: {aem_matrix_train.shape}")
print(f"   AEM term names: {aem_names}")
print(f"   Trained scaling factors:")
for name, factor in trained_scaling_factors.items():
    print(f"     {name}: {factor:.4e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Polynomial drift (linear_x + linear_y) in model space
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] Computing polynomial drift (linear_x + linear_y)...")

resc = compute_resc(SILL, x_model, y_model, RANGE)
poly_config = {"drift_terms": {"linear_x": True, "linear_y": True}}
poly_matrix_train, poly_names = compute_polynomial_drift(x_model, y_model, poly_config, resc)

print(f"   resc = {resc:.4e}")
print(f"   Polynomial drift matrix shape: {poly_matrix_train.shape}")
print(f"   Polynomial term names: {poly_names}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Combine drift matrices and build UK model
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6] Building UK model with combined drift (polynomial + AEM)...")

combined_matrix_train = np.hstack([poly_matrix_train, aem_matrix_train])
all_term_names = poly_names + aem_names

print(f"   Combined drift matrix shape: {combined_matrix_train.shape}")
print(f"   All term names: {all_term_names}")

uk_model = build_uk_model(x_model, y_model, z_obs, combined_matrix_train, vario)
print("   UK model built successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Grid prediction — reusing trained_scaling_factors
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7] Predicting on 50×50 grid (reusing trained scaling factors)...")

GRID_N = 50
gx = np.linspace(5, 95, GRID_N)
gy = np.linspace(5, 95, GRID_N)
GX, GY = np.meshgrid(gx, gy)
gx_flat = GX.ravel()
gy_flat = GY.ravel()

# Transform grid to model space
gx_model, gy_model = apply_transform(gx_flat, gy_flat, transform_params)

# Polynomial drift at grid points
poly_grid, _ = compute_polynomial_drift(gx_model, gy_model, poly_names, resc)

# AEM drift at grid points — MUST pass trained_scaling_factors
aem_grid, _, pred_scaling_factors = compute_linesink_drift_matrix(
    gx_model, gy_model,
    river_gdf,
    group_col="group",
    transform_params=transform_params,
    sill=SILL,
    strength_col="resistance",
    rescaling_method="adaptive",
    apply_anisotropy=True,
    input_scaling_factors=trained_scaling_factors   # ← critical: reuse training factors
)

combined_grid = np.hstack([poly_grid, aem_grid])

# Predict using PyKrige
from kriging import predict_at_points
z_pred_flat, var_flat = predict_at_points(uk_model, gx_model, gy_model, combined_grid)
Z_pred = z_pred_flat.reshape(GRID_N, GRID_N)
Z_var  = var_flat.reshape(GRID_N, GRID_N)

print(f"   Prediction range: [{Z_pred.min():.2f}, {Z_pred.max():.2f}]")
print(f"   Variance range:   [{Z_var.min():.4f}, {Z_var.max():.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Compute raw (unscaled) AEM potential field for visualisation
# ─────────────────────────────────────────────────────────────────────────────

print("\n[8] Computing raw AEM potential field for visualisation...")

# Raw potential for each group (before scaling)
phi_raw = {}
for group_name in aem_names:
    phi_g = np.zeros(len(gx_flat))
    segs = river_gdf[river_gdf["group"] == group_name]
    for _, row in segs.iterrows():
        coords = list(row.geometry.coords)
        for j in range(len(coords) - 1):
            p1 = np.array([coords[j][0]]), np.array([coords[j][1]])
            p2 = np.array([coords[j+1][0]]), np.array([coords[j+1][1]])
            x1m, y1m = apply_transform(p1[0], p1[1], transform_params)
            x2m, y2m = apply_transform(p2[0], p2[1], transform_params)
            phi_g += compute_linesink_potential(
                gx_model, gy_model,
                x1m[0], y1m[0], x2m[0], y2m[0],
                strength=row["resistance"]
            )
    phi_raw[group_name] = phi_g.reshape(GRID_N, GRID_N)

# Scaled potential (after applying trained_scaling_factors)
phi_scaled = {
    name: phi_raw[name] * trained_scaling_factors[name]
    for name in aem_names
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Produce multi-panel SVG figure
# ─────────────────────────────────────────────────────────────────────────────

print("\n[9] Generating figure...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "Task 4.3 — AEM Linesink Drift Example\n"
    "River influence modelled via Analytic Element Method potential",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# Helper: draw river segments on an axis
def draw_river(ax):
    colors = {"RiverA": "#1565C0", "RiverB": "#0097A7"}
    for _, row in river_gdf.iterrows():
        coords = list(row.geometry.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.plot(xs, ys, color=colors.get(row["group"], "blue"),
                linewidth=2.5, label=row["group"], zorder=5)

# ── Panel 1: Observation points + river ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sc = ax1.scatter(x_obs, y_obs, c=z_obs, cmap="viridis", s=60, zorder=6,
                 edgecolors="k", linewidths=0.5)
draw_river(ax1)
# Deduplicate legend
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right")
plt.colorbar(sc, ax=ax1, label="Water level (m)")
ax1.set_title("Observation Points\n& River Geometry", fontsize=9)
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
ax1.set_xlim(0, 100); ax1.set_ylim(0, 100)

# ── Panel 2: Raw AEM potential — RiverA ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.contourf(GX, GY, phi_raw["RiverA"], levels=20, cmap="Blues")
draw_river(ax2)
plt.colorbar(im2, ax=ax2, label="φ (raw)")
ax2.set_title("Raw AEM Potential\nRiverA (unscaled)", fontsize=9)
ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)

# ── Panel 3: Scaled AEM potential — RiverA ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.contourf(GX, GY, phi_scaled["RiverA"], levels=20, cmap="Blues")
draw_river(ax3)
plt.colorbar(im3, ax=ax3, label="φ × scale factor")
sf_A = trained_scaling_factors["RiverA"]
ax3.set_title(f"Scaled AEM Potential\nRiverA  (×{sf_A:.2e})", fontsize=9)
ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)")
ax3.set_xlim(0, 100); ax3.set_ylim(0, 100)

# ── Panel 4: Scaled AEM potential — RiverB ───────────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
im4 = ax4.contourf(GX, GY, phi_scaled["RiverB"], levels=20, cmap="Greens")
draw_river(ax4)
plt.colorbar(im4, ax=ax4, label="φ × scale factor")
sf_B = trained_scaling_factors["RiverB"]
ax4.set_title(f"Scaled AEM Potential\nRiverB  (×{sf_B:.2e})", fontsize=9)
ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)")
ax4.set_xlim(0, 100); ax4.set_ylim(0, 100)

# ── Panel 5: Drift matrix heatmap ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 0])
im5 = ax5.imshow(combined_matrix_train, aspect="auto", cmap="RdBu_r",
                 interpolation="nearest")
plt.colorbar(im5, ax=ax5, label="Drift value")
ax5.set_xticks(range(len(all_term_names)))
ax5.set_xticklabels(all_term_names, rotation=30, ha="right", fontsize=7)
ax5.set_xlabel("Drift term"); ax5.set_ylabel("Training point index")
ax5.set_title("Combined Drift Matrix\n(polynomial + AEM, training)", fontsize=9)

# ── Panel 6: UK prediction surface ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 1])
im6 = ax6.contourf(GX, GY, Z_pred, levels=20, cmap="viridis")
ax6.scatter(x_obs, y_obs, c=z_obs, cmap="viridis", s=40, edgecolors="k",
            linewidths=0.5, zorder=6)
draw_river(ax6)
plt.colorbar(im6, ax=ax6, label="Water level (m)")
ax6.set_title("UK Prediction Surface\n(polynomial + AEM drift)", fontsize=9)
ax6.set_xlabel("X (m)"); ax6.set_ylabel("Y (m)")
ax6.set_xlim(0, 100); ax6.set_ylim(0, 100)

# ── Panel 7: Kriging variance ─────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[1, 2])
im7 = ax7.contourf(GX, GY, Z_var, levels=20, cmap="YlOrRd")
ax7.scatter(x_obs, y_obs, c="white", s=20, edgecolors="k",
            linewidths=0.5, zorder=6)
draw_river(ax7)
plt.colorbar(im7, ax=ax7, label="Kriging variance")
ax7.set_title("Kriging Variance\n(lower near data)", fontsize=9)
ax7.set_xlabel("X (m)"); ax7.set_ylabel("Y (m)")
ax7.set_xlim(0, 100); ax7.set_ylim(0, 100)

# ── Panel 8: Scaling factor persistence verification ─────────────────────────
ax8 = fig.add_subplot(gs[1, 3])
ax8.axis("off")

# Verify that prediction factors match training factors exactly
factors_match = all(
    abs(pred_scaling_factors[k] - trained_scaling_factors[k]) < 1e-15
    for k in aem_names
)

lines = [
    "Scaling Factor Persistence",
    "",
    "Training (input_scaling_factors=None):",
    f"  RiverA: {trained_scaling_factors['RiverA']:.4e}",
    f"  RiverB: {trained_scaling_factors['RiverB']:.4e}",
    "",
    "Prediction (factors reused):",
    f"  RiverA: {pred_scaling_factors['RiverA']:.4e}",
    f"  RiverB: {pred_scaling_factors['RiverB']:.4e}",
    "",
    f"Factors match: {'PASS' if factors_match else 'FAIL'}",
    "",
    "Drift columns must use the",
    "same scale at training and",
    "prediction, or the solved",
    "coefficients are misapplied.",
]
ax8.text(0.05, 0.97, "\n".join(lines), transform=ax8.transAxes,
         fontsize=8, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", alpha=0.9))

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "ex_linesink_drift.svg")
fig.savefig(out_path, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"\n   Figure saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  River groups:              {aem_names}")
print(f"  Polynomial terms:          {poly_names}")
print(f"  Combined drift columns:    {len(all_term_names)}")
print(f"  Training drift shape:      {combined_matrix_train.shape}")
print(f"  Grid prediction shape:     {Z_pred.shape}")
print(f"  Scaling factors match:     {'YES' if factors_match else 'NO'}")
print(f"  Output SVG:                {out_path}")
print("=" * 70)
print("\nDone.")
