"""
Task 3.7 — V&V: Wrapper Equivalence — No Drift Path
=====================================================
Objective: Verify that build_uk_model() with an empty drift matrix produces
results equivalent to direct pykrige.uk.UniversalKriging with no specified drift.

Test cases:
  TC1: Wrapper with drift_matrix=np.zeros((20,0)) predictions match direct PyKrige
  TC2: Wrapper variances match direct PyKrige variances
  TC3: Wrapper with drift_matrix=None also matches direct PyKrige
  TC4: Wrapper with drift_matrix=np.zeros((20,0)) on a 10x10 grid matches direct PyKrige grid

Tolerances:
  Predictions: max absolute difference < 1e-10
  Variances:   max absolute difference < 1e-10

Note on predict_at_points():
  PyKrige sets uk_model.specified_drift = False (not None) when no drift is used.
  The predict_at_points() fallback check `spec_drift is not None` evaluates True
  for False, causing a false-positive drift detection.  For this V&V we call
  uk_model.execute() directly to isolate the build_uk_model() equivalence test.

Output: docs/validation/output/vv_wrapper_no_drift.svg
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []   # (tc_label, description, tolerance, max_error, status)

def check_array(label, description, arr_expected, arr_actual, tol):
    max_err = float(np.max(np.abs(arr_actual - arr_expected)))
    status = PASS if max_err <= tol else FAIL
    results.append((label, description, tol, max_err, status))
    return status, max_err


# ── minimal variogram-like object ────────────────────────────────────────────
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


# ── synthetic dataset ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

N = 20
x_train = rng.uniform(0, 100, N)
y_train = rng.uniform(0, 100, N)

# Generate z values with approximate spherical variogram structure
# Use a simple correlated random field via Cholesky decomposition
def spherical_cov(h, sill, range_, nugget):
    psill = sill - nugget
    c = np.where(
        h == 0,
        sill,
        np.where(
            h <= range_,
            psill * (1 - 1.5 * (h / range_) + 0.5 * (h / range_)**3),
            0.0
        )
    )
    return c

SILL = 1.0
RANGE = 50.0
NUGGET = 0.1

# Build covariance matrix for training points
dist_train = np.sqrt((x_train[:, None] - x_train[None, :])**2 +
                     (y_train[:, None] - y_train[None, :])**2)
C_train = spherical_cov(dist_train, SILL, RANGE, NUGGET)
# Add small jitter for numerical stability
C_train += 1e-9 * np.eye(N)
L = np.linalg.cholesky(C_train)
z_train = L @ rng.standard_normal(N)

# 10x10 prediction grid
gx = np.linspace(5, 95, 10)
gy = np.linspace(5, 95, 10)
gxx, gyy = np.meshgrid(gx, gy)
x_grid = gxx.ravel()
y_grid = gyy.ravel()

vario = SimpleVariogram(model="spherical", sill=SILL, range_=RANGE, nugget=NUGGET)
vario_params = {"sill": SILL, "range": RANGE, "nugget": NUGGET}

# ── import pykrige directly ───────────────────────────────────────────────────
try:
    from pykrige.uk import UniversalKriging
except ImportError as e:
    print(f"FATAL: pykrige not available: {e}")
    sys.exit(1)


def execute_no_drift(uk_model, x, y):
    """Call uk_model.execute() for a no-drift model, returning (z, ss) arrays."""
    res = uk_model.execute("points", np.asarray(x).ravel(), np.asarray(y).ravel())
    return np.asarray(res[0]).ravel(), np.asarray(res[1]).ravel()


# ── TC1 & TC2: point predictions — wrapper (zeros drift) vs direct PyKrige ───
print("=" * 65)
print("Task 3.7 — V&V: Wrapper Equivalence — No Drift Path")
print("=" * 65)

# Wrapper model: empty drift matrix (N x 0)
empty_drift = np.zeros((N, 0))
uk_wrapper = build_uk_model(x_train, y_train, z_train, empty_drift, vario)

# Direct PyKrige (no drift_terms kwarg at all)
uk_direct = UniversalKriging(
    x_train, y_train, z_train,
    variogram_model="spherical",
    variogram_parameters=vario_params,
    verbose=False,
    enable_plotting=False,
    anisotropy_scaling=1.0,
    anisotropy_angle=0.0,
)

# Predict at the same 10 arbitrary points
x_pts = rng.uniform(10, 90, 10)
y_pts = rng.uniform(10, 90, 10)

# Use execute() directly — see module docstring for why predict_at_points() is bypassed
z_wrap, ss_wrap = execute_no_drift(uk_wrapper, x_pts, y_pts)
z_dir, ss_dir = execute_no_drift(uk_direct, x_pts, y_pts)

status_tc1, err_tc1 = check_array(
    "TC1", "Point predictions: wrapper (zeros drift) vs direct PyKrige",
    z_dir, z_wrap, tol=1e-10
)
status_tc2, err_tc2 = check_array(
    "TC2", "Point variances: wrapper (zeros drift) vs direct PyKrige",
    ss_dir, ss_wrap, tol=1e-10
)

# ── TC3: wrapper with drift_matrix=None ───────────────────────────────────────
uk_wrapper_none = build_uk_model(x_train, y_train, z_train, None, vario)
z_wrap_none, ss_wrap_none = execute_no_drift(uk_wrapper_none, x_pts, y_pts)

status_tc3, err_tc3 = check_array(
    "TC3", "Point predictions: wrapper (None drift) vs direct PyKrige",
    z_dir, z_wrap_none, tol=1e-10
)

# ── TC4: grid predictions — wrapper vs direct PyKrige ────────────────────────
z_wrap_grid, ss_wrap_grid = execute_no_drift(uk_wrapper, x_grid, y_grid)
z_dir_grid, ss_dir_grid = execute_no_drift(uk_direct, x_grid, y_grid)

status_tc4, err_tc4 = check_array(
    "TC4", "Grid predictions (10x10): wrapper vs direct PyKrige",
    z_dir_grid, z_wrap_grid, tol=1e-10
)

# ── print summary table ───────────────────────────────────────────────────────
print(f"\n{'TC':<6} {'Description':<52} {'Tol':>10} {'MaxErr':>12} {'Status'}")
print("-" * 95)
for (label, desc, tol, err, status) in results:
    flag = "OK" if status == PASS else "!!"
    print(f"{label:<6} {desc:<52} {tol:>10.1e} {err:>12.3e}  {flag} {status}")

n_pass = sum(1 for r in results if r[4] == PASS)
n_total = len(results)
print(f"\n{'-'*95}")
print(f"Result: {n_pass}/{n_total} PASS")
overall = PASS if n_pass == n_total else FAIL
print(f"Overall: {overall}")


# ── SVG output ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#f8f9fa")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── subplot 1: training data scatter ─────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
sc = ax0.scatter(x_train, y_train, c=z_train, cmap="RdYlBu_r", s=60, edgecolors="k", linewidths=0.5, zorder=3)
plt.colorbar(sc, ax=ax0, label="z value")
ax0.set_title("Training Data (N=20, seed=42)", fontsize=9, fontweight="bold")
ax0.set_xlabel("X"); ax0.set_ylabel("Y")
ax0.set_xlim(0, 100); ax0.set_ylim(0, 100)
ax0.set_facecolor("#eef2f7")
ax0.grid(True, alpha=0.4)

# ── subplot 2: wrapper grid prediction ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
Z_wrap = z_wrap_grid.reshape(10, 10)
im1 = ax1.contourf(gxx, gyy, Z_wrap, levels=12, cmap="RdYlBu_r")
plt.colorbar(im1, ax=ax1, label="Predicted z")
ax1.scatter(x_train, y_train, c="k", s=15, zorder=3, label="Training pts")
ax1.set_title("Wrapper (empty drift)\nGrid Prediction", fontsize=9, fontweight="bold")
ax1.set_xlabel("X"); ax1.set_ylabel("Y")
ax1.set_facecolor("#eef2f7")

# ── subplot 3: direct PyKrige grid prediction ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
Z_dir = z_dir_grid.reshape(10, 10)
im2 = ax2.contourf(gxx, gyy, Z_dir, levels=12, cmap="RdYlBu_r")
plt.colorbar(im2, ax=ax2, label="Predicted z")
ax2.scatter(x_train, y_train, c="k", s=15, zorder=3, label="Training pts")
ax2.set_title("Direct PyKrige\nGrid Prediction", fontsize=9, fontweight="bold")
ax2.set_xlabel("X"); ax2.set_ylabel("Y")
ax2.set_facecolor("#eef2f7")

# ── subplot 4: absolute difference map ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
diff_grid = np.abs(z_wrap_grid - z_dir_grid).reshape(10, 10)
im3 = ax3.contourf(gxx, gyy, diff_grid, levels=12, cmap="Reds")
plt.colorbar(im3, ax=ax3, label="|Wrapper − Direct|")
ax3.set_title(f"Absolute Difference (Grid)\nMax = {err_tc4:.2e}", fontsize=9, fontweight="bold")
ax3.set_xlabel("X"); ax3.set_ylabel("Y")
ax3.set_facecolor("#eef2f7")

# ── subplot 5: point-by-point comparison scatter ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(z_dir, z_wrap, c="#2563eb", s=50, edgecolors="k", linewidths=0.5, zorder=3, label="zeros drift")
ax4.scatter(z_dir, z_wrap_none, c="#f59e0b", s=30, marker="^", edgecolors="k", linewidths=0.5, zorder=4, label="None drift")
lims = [min(z_dir.min(), z_wrap.min()) - 0.05, max(z_dir.max(), z_wrap.max()) + 0.05]
ax4.plot(lims, lims, "k--", lw=1, label="1:1 line")
ax4.set_xlim(lims); ax4.set_ylim(lims)
ax4.set_xlabel("Direct PyKrige z"); ax4.set_ylabel("Wrapper z")
ax4.set_title("Point Predictions\nWrapper vs Direct PyKrige", fontsize=9, fontweight="bold")
ax4.legend(fontsize=7)
ax4.set_facecolor("#eef2f7")
ax4.grid(True, alpha=0.4)

# ── subplot 6: pass/fail summary — text-based, 3 columns only ────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

overall_color = "#065f46" if overall == PASS else "#991b1b"
ax5.set_title(
    f"V&V Summary: {n_pass}/{n_total} PASS [{overall}]",
    fontsize=9, fontweight="bold", color=overall_color, pad=6
)

# 3-column layout: TC | Description (with max-err note) | Status
# Header bar
ax5.add_patch(plt.Rectangle((0, 0.82), 1.0, 0.13, color="#1e3a5f", zorder=2))
ax5.text(0.05, 0.885, "TC", color="white", fontsize=8, fontweight="bold", va="center", transform=ax5.transAxes)
ax5.text(0.22, 0.885, "Description", color="white", fontsize=8, fontweight="bold", va="center", transform=ax5.transAxes)
ax5.text(0.93, 0.885, "Result", color="white", fontsize=8, fontweight="bold", va="center", ha="right", transform=ax5.transAxes)

short_descs = [
    "zeros drift matrix\nvs direct PyKrige (pts)",
    "variances: zeros drift\nvs direct PyKrige (pts)",
    "None drift matrix\nvs direct PyKrige (pts)",
    "grid 10x10: zeros drift\nvs direct PyKrige",
]

row_h = 0.175
for i, (label, desc, tol, err, status) in enumerate(results):
    y_top = 0.82 - i * row_h
    y_mid = y_top - row_h / 2
    bg = "#d1fae5" if status == PASS else "#fee2e2"
    ax5.add_patch(plt.Rectangle((0, y_top - row_h), 1.0, row_h, color=bg, zorder=1))
    ax5.axhline(y_top, color="#aaa", lw=0.5, zorder=3)
    sd = short_descs[i] if i < len(short_descs) else desc[:28]
    ax5.text(0.05, y_mid, label, fontsize=7.5, va="center", fontweight="bold", transform=ax5.transAxes)
    ax5.text(0.22, y_mid, sd, fontsize=6.5, va="center", transform=ax5.transAxes, linespacing=1.3)
    fc = "#065f46" if status == PASS else "#991b1b"
    ax5.text(0.93, y_mid, status, fontsize=8, va="center", ha="right", color=fc, fontweight="bold", transform=ax5.transAxes)

fig.suptitle(
    "Task 3.7 — Wrapper Equivalence: No Drift Path\n"
    "build_uk_model(empty drift) ≡ direct UniversalKriging",
    fontsize=11, fontweight="bold", y=0.98
)

svg_path = os.path.join(OUTPUT_DIR, "vv_wrapper_no_drift.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=150)
png_path = os.path.join(OUTPUT_DIR, "vv_wrapper_no_drift.png")
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSVG saved -> {svg_path}")
