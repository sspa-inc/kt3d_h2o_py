"""
Task 4.2 — EXAMPLE: Universal Kriging with Linear Drift
========================================================
Demonstrates modeling a regional groundwater gradient using Universal Kriging
with a linear_x drift term.

Scenario:
  - 30 synthetic observation points with z = 0.5*x + noise
  - Enable linear_x drift term
  - Show the drift matrix and explain what each column represents
  - Compare prediction with and without drift enabled
  - Show verify_drift_physics() output and explain PASS/FAIL

Outputs:
  docs/examples/output/ex_linear_drift.svg   (4-panel comparison figure)

Run from the project root:
    python docs/examples/ex_linear_drift.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kriging import build_uk_model
from drift import compute_resc, compute_polynomial_drift, verify_drift_physics

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("Task 4.2 — EXAMPLE: Universal Kriging with Linear Drift")
print("=" * 65)

# ── Variogram stub ────────────────────────────────────────────────────────────
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


SILL   = 1.0
RANGE  = 50.0
NUGGET = 0.1

vario = SimpleVariogram(model="spherical", sill=SILL, range_=RANGE, nugget=NUGGET)

print(f"\nVariogram: spherical  sill={SILL}  range={RANGE}  nugget={NUGGET}")

# ── 1. Synthetic training data ────────────────────────────────────────────────
# True field: z = 0.5*x + N(0, 0.3)
# This represents a regional groundwater gradient declining from west to east.
# The linear_x drift term is designed to capture exactly this kind of trend.

rng = np.random.default_rng(42)
N = 30
x_train = rng.uniform(0, 100, N)
y_train = rng.uniform(0, 100, N)

GRADIENT = 0.5          # true linear coefficient (m/m or m/km depending on units)
NOISE_STD = 0.3         # residual noise standard deviation
z_true = GRADIENT * x_train
z_train = z_true + rng.normal(0, NOISE_STD, N)

print(f"\nTrue field: z = {GRADIENT}*x + N(0, {NOISE_STD})")
print(f"Training data: N={N} points")
print(f"  x range: [{x_train.min():.1f}, {x_train.max():.1f}]")
print(f"  y range: [{y_train.min():.1f}, {y_train.max():.1f}]")
print(f"  z range: [{z_train.min():.3f}, {z_train.max():.3f}]")

# ── 2. Compute rescaling factor and drift matrix ──────────────────────────────
# The rescaling factor normalises drift columns so they are numerically
# comparable to the variogram sill.  Formula:
#   resc = sqrt(sill / max(radsqd, range²))
# where radsqd = max squared distance from centroid to any training point.

resc = compute_resc(SILL, x_train, y_train, RANGE)
print(f"\nRescaling factor: resc = {resc:.6e}")
print(f"  (sqrt({SILL} / max(radsqd, {RANGE}²)))")

# Config dict enabling only linear_x
drift_config = {"drift_terms": {"linear_x": True}}

drift_matrix, term_names = compute_polynomial_drift(x_train, y_train, drift_config, resc)

print(f"\nDrift matrix shape: {drift_matrix.shape}  (N_points × N_terms)")
print(f"Term names: {term_names}")
print("\nDrift matrix — first 5 rows:")
print(f"  {'x':>8}  {'linear_x drift':>16}  {'= resc * x':>12}")
print(f"  {'--------':>8}  {'----------------':>16}  {'----------':>12}")
for i in range(min(5, N)):
    print(f"  {x_train[i]:8.3f}  {drift_matrix[i, 0]:16.6e}  {resc * x_train[i]:12.6e}")

# ── 3. Verify drift physics ───────────────────────────────────────────────────
print("\n--- verify_drift_physics() ---")
physics_results = verify_drift_physics(drift_matrix, term_names, x_train, y_train, resc)
for term, result in physics_results.items():
    print(f"  {term}: {result}")
print("  (PASS = R² > 0.999 and slope within 1% of resc)")

# ── 4. Build models: with drift and without drift ─────────────────────────────
print("\nBuilding UK model WITH linear_x drift...")
uk_with_drift = build_uk_model(
    x_train, y_train, z_train,
    drift_matrix=drift_matrix,
    variogram=vario
)
print("  Model built successfully.")

print("\nBuilding OK model WITHOUT drift (for comparison)...")
uk_no_drift = build_uk_model(
    x_train, y_train, z_train,
    drift_matrix=None,
    variogram=vario
)
print("  Model built successfully.")

# ── 5. Predict on a 50x50 grid ────────────────────────────────────────────────
GRID_RES = 50
gx = np.linspace(0, 100, GRID_RES)
gy = np.linspace(0, 100, GRID_RES)
GX, GY = np.meshgrid(gx, gy)
x_grid = GX.ravel()
y_grid = GY.ravel()

# Drift at grid nodes (must use same resc as training)
drift_grid, _ = compute_polynomial_drift(x_grid, y_grid, drift_config, resc)

print(f"\nPredicting on {GRID_RES}x{GRID_RES} grid ({len(x_grid)} nodes)...")

# With drift: pass drift columns to execute()
drift_cols_grid = [drift_grid[:, i] for i in range(drift_grid.shape[1])]
res_drift = uk_with_drift.execute("points", x_grid, y_grid,
                                   specified_drift_arrays=drift_cols_grid)
z_with = np.asarray(res_drift[0]).reshape(GRID_RES, GRID_RES)
v_with = np.asarray(res_drift[1]).reshape(GRID_RES, GRID_RES)

# Without drift: ordinary kriging execute
res_nodrift = uk_no_drift.execute("points", x_grid, y_grid)
z_without = np.asarray(res_nodrift[0]).reshape(GRID_RES, GRID_RES)
v_without = np.asarray(res_nodrift[1]).reshape(GRID_RES, GRID_RES)

# True trend on grid
z_true_grid = (GRADIENT * GX).reshape(GRID_RES, GRID_RES)

print(f"  With drift    — prediction range: [{z_with.min():.3f}, {z_with.max():.3f}]")
print(f"  Without drift — prediction range: [{z_without.min():.3f}, {z_without.max():.3f}]")
print(f"  True trend    — range:            [{z_true_grid.min():.3f}, {z_true_grid.max():.3f}]")

# ── 6. Generate SVG output ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Universal Kriging with Linear Drift  (z = 0.5·x + noise,  linear_x enabled)",
    fontsize=12, fontweight="bold"
)

gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.45)

# Shared colour limits for prediction panels
vmin_pred = min(z_with.min(), z_without.min(), z_true_grid.min())
vmax_pred = max(z_with.max(), z_without.max(), z_true_grid.max())

# ── Panel 1: True trend ───────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
cf0 = ax0.contourf(GX, GY, z_true_grid, levels=20, cmap="RdYlBu_r",
                   vmin=vmin_pred, vmax=vmax_pred)
ax0.scatter(x_train, y_train, c=z_train, cmap="RdYlBu_r",
            edgecolors="k", linewidths=0.7, s=50, zorder=5,
            vmin=vmin_pred, vmax=vmax_pred)
plt.colorbar(cf0, ax=ax0, label="z")
ax0.set_title("True Trend  (z = 0.5·x)", fontsize=9)
ax0.set_xlabel("X"); ax0.set_ylabel("Y")
ax0.set_aspect("equal")

# ── Panel 2: Prediction WITH drift ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
cf1 = ax1.contourf(GX, GY, z_with, levels=20, cmap="RdYlBu_r",
                   vmin=vmin_pred, vmax=vmax_pred)
ax1.scatter(x_train, y_train, c=z_train, cmap="RdYlBu_r",
            edgecolors="k", linewidths=0.7, s=50, zorder=5,
            vmin=vmin_pred, vmax=vmax_pred)
plt.colorbar(cf1, ax=ax1, label="z")
ax1.set_title("UK WITH linear_x drift", fontsize=9)
ax1.set_xlabel("X"); ax1.set_ylabel("Y")
ax1.set_aspect("equal")

# ── Panel 3: Prediction WITHOUT drift ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
cf2 = ax2.contourf(GX, GY, z_without, levels=20, cmap="RdYlBu_r",
                   vmin=vmin_pred, vmax=vmax_pred)
ax2.scatter(x_train, y_train, c=z_train, cmap="RdYlBu_r",
            edgecolors="k", linewidths=0.7, s=50, zorder=5,
            vmin=vmin_pred, vmax=vmax_pred)
plt.colorbar(cf2, ax=ax2, label="z")
ax2.set_title("OK WITHOUT drift", fontsize=9)
ax2.set_xlabel("X"); ax2.set_ylabel("Y")
ax2.set_aspect("equal")

# ── Panel 4: Kriging variance WITH drift ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
cf3 = ax3.contourf(GX, GY, v_with, levels=20, cmap="YlOrRd")
ax3.scatter(x_train, y_train, c="white", edgecolors="k",
            linewidths=0.7, s=50, zorder=5)
plt.colorbar(cf3, ax=ax3, label="Variance")
ax3.set_title("Variance — WITH drift", fontsize=9)
ax3.set_xlabel("X"); ax3.set_ylabel("Y")
ax3.set_aspect("equal")

# ── Panel 5: Kriging variance WITHOUT drift ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
cf4 = ax4.contourf(GX, GY, v_without, levels=20, cmap="YlOrRd")
ax4.scatter(x_train, y_train, c="white", edgecolors="k",
            linewidths=0.7, s=50, zorder=5)
plt.colorbar(cf4, ax=ax4, label="Variance")
ax4.set_title("Variance — WITHOUT drift", fontsize=9)
ax4.set_xlabel("X"); ax4.set_ylabel("Y")
ax4.set_aspect("equal")

# ── Panel 6: Drift column visualisation ──────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
# Show drift column value vs x coordinate
ax5.scatter(x_train, drift_matrix[:, 0], c="steelblue", edgecolors="k",
            linewidths=0.7, s=60, zorder=5, label="drift = resc·x")
x_line = np.linspace(0, 100, 200)
ax5.plot(x_line, resc * x_line, "r--", linewidth=1.5, label=f"resc·x  (resc={resc:.4f})")
ax5.set_xlabel("X coordinate")
ax5.set_ylabel("Drift column value")
ax5.set_title("Drift Matrix Column: linear_x", fontsize=9)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Physics verification annotation
phys_str = "\n".join(
    [f"verify_drift_physics():"] +
    [f"  {t}: {r}" for t, r in physics_results.items()]
)
ax5.text(0.03, 0.97, phys_str, transform=ax5.transAxes,
         fontsize=7.5, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# Footer annotation
info = (
    f"N = {N} training points  |  True field: z = {GRADIENT}·x + N(0, {NOISE_STD})\n"
    f"Variogram: spherical  sill={SILL}  range={RANGE}  nugget={NUGGET}  |  "
    f"resc = {resc:.4e}  |  Grid: {GRID_RES}×{GRID_RES}"
)
fig.text(0.5, 0.01, info, ha="center", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

svg_path = os.path.join(OUTPUT_DIR, "ex_linear_drift.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"\nSVG saved -> {svg_path}")

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Example complete.")
print(f"  Training points : {N}")
print(f"  True gradient   : z = {GRADIENT}*x + N(0, {NOISE_STD})")
print(f"  Drift terms     : linear_x  (resc={resc:.4e})")
print(f"  Drift matrix    : shape {drift_matrix.shape}")
print(f"  Physics check   : {physics_results}")
print(f"  Grid nodes      : {GRID_RES}x{GRID_RES} = {GRID_RES**2}")
print(f"  Output SVG      : {svg_path}")
print("=" * 65)
