"""
Task 3.8 — V&V: Specified Polynomial Drift — Synthetic Truth Recovery
======================================================================
Objective: Verify that Universal Kriging with polynomial drift can recover
a known linear trend from synthetic data.

Setup:
  - 30 points (seed=42) with z = 0.5*x + 0.3*y + N(0, 0.1)
  - Variogram: spherical, sill=0.1, range=50, nugget=0.01

Test cases:
  TC1: Our wrapper predictions match direct PyKrige specified-drift (max |diff| < 1e-8)
  TC2: Our wrapper variances match direct PyKrige specified-drift (max |diff| < 1e-8)
  TC3: Trend recovery RMSE < 0.5 (noise-limited)
  TC4: Direct PyKrige trend recovery RMSE < 0.5

Output: docs/validation/output/vv_polynomial_drift_recovery.svg
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kriging import build_uk_model
from drift import compute_resc, compute_polynomial_drift

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []   # (tc_label, description, tolerance, metric_value, status)


def check_metric(label, description, value, tol, lower_is_better=True):
    """Record a pass/fail result for a scalar metric."""
    if lower_is_better:
        status = PASS if value <= tol else FAIL
    else:
        status = PASS if value >= tol else FAIL
    results.append((label, description, tol, value, status))
    return status, value


# ── minimal variogram-like object ────────────────────────────────────────────
class SimpleVariogram:
    """Minimal variogram object compatible with build_uk_model()."""
    def __init__(self, model="spherical", sill=0.1, range_=50.0, nugget=0.01):
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


# ── synthetic dataset ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

N = 30
x_train = rng.uniform(0, 100, N)
y_train = rng.uniform(0, 100, N)

SILL   = 0.1
RANGE  = 50.0
NUGGET = 0.01

# True trend: z = 0.5*x + 0.3*y + noise
noise = rng.normal(0, 0.1, N)
z_train = 0.5 * x_train + 0.3 * y_train + noise

# ── prediction grid ───────────────────────────────────────────────────────────
gx = np.linspace(5, 95, 20)
gy = np.linspace(5, 95, 20)
gxx, gyy = np.meshgrid(gx, gy)
x_grid = gxx.ravel()
y_grid = gyy.ravel()

# True trend on grid
z_true_grid = 0.5 * x_grid + 0.3 * y_grid

# ── variogram object ──────────────────────────────────────────────────────────
vario = SimpleVariogram(model="spherical", sill=SILL, range_=RANGE, nugget=NUGGET)
vario_params = {"sill": SILL, "range": RANGE, "nugget": NUGGET}

# ── import pykrige directly ───────────────────────────────────────────────────
try:
    from pykrige.uk import UniversalKriging
except ImportError as e:
    print(f"FATAL: pykrige not available: {e}")
    sys.exit(1)

# ── compute drift (training points) ──────────────────────────────────────────
# Config: linear_x + linear_y only
drift_config = {"drift_terms": {"linear_x": True, "linear_y": True,
                                "quadratic_x": False, "quadratic_y": False}}

resc = compute_resc(SILL, x_train, y_train, RANGE)
drift_train, term_names = compute_polynomial_drift(x_train, y_train, drift_config, resc)
# drift_train shape: (N, 2)  columns: [linear_x, linear_y]

# ── compute drift (grid points) ───────────────────────────────────────────────
drift_grid, _ = compute_polynomial_drift(x_grid, y_grid, term_names, resc)
# drift_grid shape: (N_grid, 2)

# ── build our wrapper model ───────────────────────────────────────────────────
print("=" * 65)
print("Task 3.8 — V&V: Polynomial Drift Recovery")
print("=" * 65)
print(f"\nresc = {resc:.6e}")
print(f"term_names = {term_names}")
print(f"drift_train shape = {drift_train.shape}")

uk_wrapper = build_uk_model(x_train, y_train, z_train, drift_train, vario)

# Predict on grid using execute() with specified drift columns
drift_grid_cols = [drift_grid[:, i].ravel() for i in range(drift_grid.shape[1])]
res_wrap = uk_wrapper.execute("points", x_grid, y_grid,
                               specified_drift_arrays=drift_grid_cols)
z_wrap_grid = np.asarray(res_wrap[0]).ravel()
ss_wrap_grid = np.asarray(res_wrap[1]).ravel()

# ── build direct PyKrige model with specified drift ───────────────────────────
# Pass the same drift columns as specified_drift at training time
drift_train_cols = [drift_train[:, i].ravel() for i in range(drift_train.shape[1])]

uk_direct = UniversalKriging(
    x_train, y_train, z_train,
    variogram_model="spherical",
    variogram_parameters=vario_params,
    drift_terms=["specified"],
    specified_drift=drift_train_cols,
    verbose=False,
    enable_plotting=False,
    anisotropy_scaling=1.0,
    anisotropy_angle=0.0,
)

res_dir = uk_direct.execute("points", x_grid, y_grid,
                             specified_drift_arrays=drift_grid_cols)
z_dir_grid = np.asarray(res_dir[0]).ravel()
ss_dir_grid = np.asarray(res_dir[1]).ravel()

# ── TC1: wrapper vs direct PyKrige predictions ────────────────────────────────
max_diff_pred = float(np.max(np.abs(z_wrap_grid - z_dir_grid)))
status_tc1, _ = check_metric(
    "TC1",
    "Wrapper vs direct PyKrige: max |pred diff|",
    max_diff_pred, tol=1e-8
)

# ── TC2: wrapper vs direct PyKrige variances ──────────────────────────────────
max_diff_var = float(np.max(np.abs(ss_wrap_grid - ss_dir_grid)))
status_tc2, _ = check_metric(
    "TC2",
    "Wrapper vs direct PyKrige: max |var diff|",
    max_diff_var, tol=1e-8
)

# ── TC3: trend recovery RMSE — our wrapper ────────────────────────────────────
rmse_wrap = float(np.sqrt(np.mean((z_wrap_grid - z_true_grid) ** 2)))
status_tc3, _ = check_metric(
    "TC3",
    "Wrapper trend recovery RMSE vs 0.5x+0.3y",
    rmse_wrap, tol=0.5
)

# ── TC4: trend recovery RMSE — direct PyKrige ────────────────────────────────
rmse_dir = float(np.sqrt(np.mean((z_dir_grid - z_true_grid) ** 2)))
status_tc4, _ = check_metric(
    "TC4",
    "Direct PyKrige trend recovery RMSE vs 0.5x+0.3y",
    rmse_dir, tol=0.5
)

# ── print summary table ───────────────────────────────────────────────────────
print(f"\n{'TC':<6} {'Description':<50} {'Tol':>10} {'Value':>12} {'Status'}")
print("-" * 90)
for (label, desc, tol, val, status) in results:
    flag = "OK" if status == PASS else "!!"
    print(f"{label:<6} {desc:<50} {tol:>10.1e} {val:>12.3e}  {flag} {status}")

n_pass = sum(1 for r in results if r[4] == PASS)
n_total = len(results)
print(f"\n{'-'*90}")
print(f"Result: {n_pass}/{n_total} PASS")
overall = PASS if n_pass == n_total else FAIL
print(f"Overall: {overall}")

# ── SVG output ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#f8f9fa")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── subplot 1: training data scatter ─────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
sc = ax0.scatter(x_train, y_train, c=z_train, cmap="RdYlBu_r", s=60,
                 edgecolors="k", linewidths=0.5, zorder=3)
plt.colorbar(sc, ax=ax0, label="z = 0.5x + 0.3y + noise")
ax0.set_title("Training Data (N=30, seed=42)\nz = 0.5x + 0.3y + N(0,0.1)",
              fontsize=9, fontweight="bold")
ax0.set_xlabel("X"); ax0.set_ylabel("Y")
ax0.set_xlim(0, 100); ax0.set_ylim(0, 100)
ax0.set_facecolor("#eef2f7")
ax0.grid(True, alpha=0.4)

# ── subplot 2: wrapper predicted surface ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
Z_wrap = z_wrap_grid.reshape(20, 20)
levels = np.linspace(z_wrap_grid.min(), z_wrap_grid.max(), 14)
im1 = ax1.contourf(gxx, gyy, Z_wrap, levels=levels, cmap="RdYlBu_r")
plt.colorbar(im1, ax=ax1, label="Predicted z")
ax1.scatter(x_train, y_train, c="k", s=15, zorder=3, label="Training pts")
ax1.set_title("Wrapper UK Prediction\n(linear_x + linear_y drift)",
              fontsize=9, fontweight="bold")
ax1.set_xlabel("X"); ax1.set_ylabel("Y")
ax1.set_facecolor("#eef2f7")
ax1.grid(True, alpha=0.3)

# ── subplot 3: true trend surface ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
Z_true = z_true_grid.reshape(20, 20)
im2 = ax2.contourf(gxx, gyy, Z_true, levels=levels, cmap="RdYlBu_r")
plt.colorbar(im2, ax=ax2, label="True z")
ax2.scatter(x_train, y_train, c="k", s=15, zorder=3, label="Training pts")
ax2.set_title("True Trend Surface\nz = 0.5x + 0.3y",
              fontsize=9, fontweight="bold")
ax2.set_xlabel("X"); ax2.set_ylabel("Y")
ax2.set_facecolor("#eef2f7")
ax2.grid(True, alpha=0.3)

# ── subplot 4: wrapper vs direct PyKrige difference ───────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
diff_grid = np.abs(z_wrap_grid - z_dir_grid).reshape(20, 20)
im3 = ax3.contourf(gxx, gyy, diff_grid, levels=14, cmap="Reds")
plt.colorbar(im3, ax=ax3, label="|Wrapper − Direct PyKrige|")
ax3.set_title(f"Wrapper vs Direct PyKrige\nMax |diff| = {max_diff_pred:.2e}",
              fontsize=9, fontweight="bold")
ax3.set_xlabel("X"); ax3.set_ylabel("Y")
ax3.set_facecolor("#eef2f7")
ax3.grid(True, alpha=0.3)

# ── subplot 5: residual from true trend ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
resid_grid = (z_wrap_grid - z_true_grid).reshape(20, 20)
vmax = max(abs(resid_grid.min()), abs(resid_grid.max()))
im4 = ax4.contourf(gxx, gyy, resid_grid, levels=14, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
plt.colorbar(im4, ax=ax4, label="Predicted − True")
ax4.set_title(f"Residual from True Trend\nRMSE = {rmse_wrap:.4f}",
              fontsize=9, fontweight="bold")
ax4.set_xlabel("X"); ax4.set_ylabel("Y")
ax4.set_facecolor("#eef2f7")
ax4.grid(True, alpha=0.3)

# ── subplot 6: pass/fail summary ─────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

overall_color = "#065f46" if overall == PASS else "#991b1b"
ax5.set_title(
    f"V&V Summary: {n_pass}/{n_total} PASS [{overall}]",
    fontsize=9, fontweight="bold", color=overall_color, pad=6
)

# Header bar
ax5.add_patch(plt.Rectangle((0, 0.82), 1.0, 0.13, color="#1e3a5f", zorder=2))
ax5.text(0.05, 0.885, "TC", color="white", fontsize=8, fontweight="bold",
         va="center", transform=ax5.transAxes)
ax5.text(0.22, 0.885, "Description", color="white", fontsize=8, fontweight="bold",
         va="center", transform=ax5.transAxes)
ax5.text(0.93, 0.885, "Result", color="white", fontsize=8, fontweight="bold",
         va="center", ha="right", transform=ax5.transAxes)

short_descs = [
    "Wrapper vs direct PyKrige\nmax |pred diff| < 1e-8",
    "Wrapper vs direct PyKrige\nmax |var diff| < 1e-8",
    f"Wrapper RMSE vs true trend\n({rmse_wrap:.4f} < 0.5)",
    f"Direct PyKrige RMSE vs true\n({rmse_dir:.4f} < 0.5)",
]

row_h = 0.175
for i, (label, desc, tol, val, status) in enumerate(results):
    y_top = 0.82 - i * row_h
    y_mid = y_top - row_h / 2
    bg = "#d1fae5" if status == PASS else "#fee2e2"
    ax5.add_patch(plt.Rectangle((0, y_top - row_h), 1.0, row_h, color=bg, zorder=1))
    ax5.axhline(y_top, color="#aaa", lw=0.5, zorder=3)
    sd = short_descs[i] if i < len(short_descs) else desc[:28]
    ax5.text(0.05, y_mid, label, fontsize=7.5, va="center", fontweight="bold",
             transform=ax5.transAxes)
    ax5.text(0.22, y_mid, sd, fontsize=6.5, va="center", transform=ax5.transAxes,
             linespacing=1.3)
    fc = "#065f46" if status == PASS else "#991b1b"
    ax5.text(0.93, y_mid, status, fontsize=8, va="center", ha="right",
             color=fc, fontweight="bold", transform=ax5.transAxes)

fig.suptitle(
    "Task 3.8 — Polynomial Drift Recovery: Synthetic Truth\n"
    "UK with linear_x + linear_y drift recovers z = 0.5x + 0.3y",
    fontsize=11, fontweight="bold", y=0.98
)

svg_path = os.path.join(OUTPUT_DIR, "vv_polynomial_drift_recovery.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=150)
png_path = os.path.join(OUTPUT_DIR, "vv_polynomial_drift_recovery.png")
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSVG saved -> {svg_path}")
