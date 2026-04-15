"""
Task 4.1 — EXAMPLE: Ordinary Kriging (No Drift)
================================================
Demonstrates the simplest use case of the UK_SSPA v2 kriging pipeline:

  - 20 synthetic observation points with spatial correlation
  - Spherical variogram, no anisotropy, no drift terms
  - Predict on a 50x50 grid
  - Generate a prediction map and variance map saved as SVG

This example uses the project's build_uk_model() wrapper directly, bypassing
the full config-driven main.py pipeline, to show the core API clearly.

Outputs:
  docs/examples/output/ex_ordinary_kriging.svg   (prediction + variance maps)

Run from the project root:
    python docs/examples/ex_ordinary_kriging.py
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Variogram configuration ───────────────────────────────────────────────
# Minimal config equivalent to:
#   {
#     "variogram": {"model": "spherical", "sill": 1.0, "range": 50, "nugget": 0.1},
#     "drift_terms": {},
#     "grid": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100, "resolution": 2}
#   }

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
PSILL  = SILL - NUGGET

vario = SimpleVariogram(model="spherical", sill=SILL, range_=RANGE, nugget=NUGGET)

print("=" * 65)
print("Task 4.1 — EXAMPLE: Ordinary Kriging (No Drift)")
print("=" * 65)
print(f"\nVariogram: spherical  sill={SILL}  range={RANGE}  nugget={NUGGET}")

# ── 2. Synthetic training data ────────────────────────────────────────────────
# Generate 20 spatially correlated points using Cholesky decomposition so the
# data has the same covariance structure as the specified variogram.

rng = np.random.default_rng(42)
N = 20
x_train = rng.uniform(0, 100, N)
y_train = rng.uniform(0, 100, N)

def spherical_cov(h, sill, range_, nugget):
    """Spherical covariance function (C(h) = sill - gamma(h))."""
    psill = sill - nugget
    c = np.where(
        h == 0,
        sill,
        np.where(
            h <= range_,
            psill * (1.0 - 1.5 * (h / range_) + 0.5 * (h / range_)**3),
            0.0
        )
    )
    return c

dist_train = np.sqrt(
    (x_train[:, None] - x_train[None, :])**2 +
    (y_train[:, None] - y_train[None, :])**2
)
C_train = spherical_cov(dist_train, SILL, RANGE, NUGGET)
C_train += 1e-9 * np.eye(N)   # jitter for numerical stability
L = np.linalg.cholesky(C_train)
z_train = L @ rng.standard_normal(N)

print(f"\nTraining data: N={N} points")
print(f"  x range: [{x_train.min():.1f}, {x_train.max():.1f}]")
print(f"  y range: [{y_train.min():.1f}, {y_train.max():.1f}]")
print(f"  z range: [{z_train.min():.3f}, {z_train.max():.3f}]")

# ── 3. Build the ordinary kriging model ───────────────────────────────────────
# Passing drift_matrix=None (or np.zeros((N,0))) triggers the ordinary kriging
# path — no drift_terms kwarg is passed to PyKrige.

print("\nBuilding ordinary kriging model (no drift)...")
uk_model = build_uk_model(
    x_train, y_train, z_train,
    drift_matrix=None,   # <-- ordinary kriging: no drift
    variogram=vario
)
print("  Model built successfully.")

# ── 4. Predict on a 50x50 grid ────────────────────────────────────────────────
GRID_RES = 50
gx = np.linspace(0, 100, GRID_RES)
gy = np.linspace(0, 100, GRID_RES)
GX, GY = np.meshgrid(gx, gy)
x_grid = GX.ravel()
y_grid = GY.ravel()

print(f"\nPredicting on {GRID_RES}x{GRID_RES} grid ({len(x_grid)} nodes)...")
res = uk_model.execute("points", x_grid, y_grid)
z_pred = np.asarray(res[0]).reshape(GRID_RES, GRID_RES)
z_var  = np.asarray(res[1]).reshape(GRID_RES, GRID_RES)

print(f"  Prediction range: [{z_pred.min():.3f}, {z_pred.max():.3f}]")
print(f"  Variance range:   [{z_var.min():.4f}, {z_var.max():.4f}]")

# ── 5. Generate SVG output ────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.5))
fig.suptitle(
    "Ordinary Kriging — No Drift  (spherical variogram, sill=1.0, range=50, nugget=0.1)",
    fontsize=11, fontweight="bold"
)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Left panel: prediction surface ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
cf1 = ax1.contourf(GX, GY, z_pred, levels=20, cmap="RdYlBu_r")
ax1.scatter(x_train, y_train, c=z_train, cmap="RdYlBu_r",
            edgecolors="k", linewidths=0.8, s=60, zorder=5,
            vmin=z_pred.min(), vmax=z_pred.max(), label="Observations")
plt.colorbar(cf1, ax=ax1, label="Predicted value")
ax1.set_title("Prediction Surface", fontsize=10)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_aspect("equal")

# ── Right panel: kriging variance ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
cf2 = ax2.contourf(GX, GY, z_var, levels=20, cmap="YlOrRd")
ax2.scatter(x_train, y_train, c="white", edgecolors="k",
            linewidths=0.8, s=60, zorder=5, label="Observations")
plt.colorbar(cf2, ax=ax2, label="Kriging variance")
ax2.set_title("Kriging Variance", fontsize=10)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend(loc="upper right", fontsize=8)
ax2.set_aspect("equal")

# Annotation box summarising the setup
info = (
    f"N = {N} training points  |  Variogram: spherical\n"
    f"sill = {SILL}  range = {RANGE}  nugget = {NUGGET}\n"
    f"Grid: {GRID_RES}×{GRID_RES}  |  No drift terms  |  No anisotropy"
)
fig.text(0.5, 0.01, info, ha="center", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

svg_path = os.path.join(OUTPUT_DIR, "ex_ordinary_kriging.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"\nSVG saved -> {svg_path}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Example complete.")
print(f"  Training points : {N}")
print(f"  Grid nodes      : {GRID_RES}x{GRID_RES} = {GRID_RES**2}")
print(f"  Drift terms     : none (ordinary kriging)")
print(f"  Anisotropy      : disabled")
print(f"  Output SVG      : {svg_path}")
print("=" * 65)
