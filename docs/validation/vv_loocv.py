"""
Task 3.10 — V&V: LOOCV Diagnostic Metrics
==========================================
Objective: Verify that cross_validate() produces correct leave-one-out statistics.

Setup:
  - 15 points (seed=42), simple linear trend + noise
  - No drift (ordinary kriging path) for simplicity

Test cases:
  TC1: n predictions are produced for n input points
  TC2: Each prediction excludes the held-out point (prediction at held-out
       location differs from observed value when nugget > 0)
  TC3: RMSE calculation: sqrt(mean((pred - obs)²))
  TC4: MAE calculation: mean(|pred - obs|)
  TC5: Q1 (mean standardized error) and Q2 (variance of standardized errors)
       are computed and finite
  TC6: With < 3 points, function returns NaN metrics gracefully

Tolerances:
  Metric calculations exact to < 1e-12.

Output: docs/validation/output/vv_loocv.svg
"""

import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARENT = os.path.abspath(os.path.join(ROOT, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Also add parent so that `from v2_Code.drift import ...` inside kriging.py resolves
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from kriging import cross_validate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []   # (tc_label, description, tolerance, value, status)


def record(label, description, tol, value, status):
    results.append((label, description, tol, value, status))


def check_scalar(label, description, expected, actual, tol):
    err = abs(actual - expected)
    status = PASS if err <= tol else FAIL
    record(label, description, tol, err, status)
    return status, err


def check_true(label, description, condition):
    status = PASS if condition else FAIL
    record(label, description, None, None, status)
    return status


# ── minimal variogram-like object ────────────────────────────────────────────
class SimpleVariogram:
    """Minimal variogram object compatible with cross_validate()."""
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

N = 15
x_all = rng.uniform(0, 100, N)
y_all = rng.uniform(0, 100, N)
# Simple linear trend + noise (no drift terms used in config -> ordinary kriging)
z_all = 0.5 * x_all + 0.3 * y_all + rng.normal(0, 0.5, N)

vario = SimpleVariogram(model="spherical", sill=1.0, range_=50.0, nugget=0.1)

# Config with NO drift terms (ordinary kriging path)
cfg_no_drift = {
    "drift_terms": {
        "linear_x": False,
        "linear_y": False,
        "quadratic_x": False,
        "quadratic_y": False,
    }
}

# ── run cross_validate ────────────────────────────────────────────────────────
print("Running cross_validate() on 15-point dataset (no drift)...")
cv_result = cross_validate(x_all, y_all, z_all, cfg_no_drift, vario)

preds = cv_result["predictions"]
vars_ = cv_result["variances"]
obs   = cv_result["observations"]

# ── TC1: n predictions produced for n input points ────────────────────────────
tc1_status = check_true(
    "TC1",
    "n predictions produced for n input points",
    len(preds) == N
)
print(f"TC1 [{tc1_status}]: len(predictions)={len(preds)}, expected {N}")

# ── TC2: each prediction differs from observed (nugget > 0 -> interpolation != exact)
# With nugget > 0, LOOCV predictions at held-out locations should NOT exactly
# reproduce the observed value (the model smooths through the data).
# We verify that at least 80% of predictions differ from observations by > 1e-6.
diffs = np.abs(preds - obs)
frac_different = np.mean(diffs > 1e-6)
tc2_status = check_true(
    "TC2",
    "Predictions differ from observations (nugget > 0, no exact interpolation)",
    frac_different >= 0.8
)
print(f"TC2 [{tc2_status}]: fraction of predictions differing from obs by >1e-6 = {frac_different:.2f} (need >=0.80)")

# ── TC3: RMSE calculation ─────────────────────────────────────────────────────
# Recompute RMSE manually from the returned arrays (using valid mask)
valid_mask = ~np.isnan(preds)
errors = preds[valid_mask] - obs[valid_mask]
rmse_manual = float(np.sqrt(np.mean(errors ** 2)))
tc3_status, tc3_err = check_scalar(
    "TC3",
    "RMSE = sqrt(mean((pred - obs)^2))",
    rmse_manual,
    cv_result["rmse"],
    1e-12
)
print(f"TC3 [{tc3_status}]: RMSE reported={cv_result['rmse']:.8f}, manual={rmse_manual:.8f}, |err|={tc3_err:.2e}")

# ── TC4: MAE calculation ──────────────────────────────────────────────────────
mae_manual = float(np.mean(np.abs(errors)))
tc4_status, tc4_err = check_scalar(
    "TC4",
    "MAE = mean(|pred - obs|)",
    mae_manual,
    cv_result["mae"],
    1e-12
)
print(f"TC4 [{tc4_status}]: MAE reported={cv_result['mae']:.8f}, manual={mae_manual:.8f}, |err|={tc4_err:.2e}")

# ── TC5: Q1 and Q2 are finite ─────────────────────────────────────────────────
q1_finite = not math.isnan(cv_result["q1"])
q2_finite = not math.isnan(cv_result["q2"])
tc5_status = check_true(
    "TC5",
    "Q1 (mean standardized error) and Q2 (variance of standardized errors) are finite",
    q1_finite and q2_finite
)
print(f"TC5 [{tc5_status}]: Q1={cv_result['q1']:.6f}, Q2={cv_result['q2']:.6f}")

# Also verify Q1/Q2 manually
var_vals = vars_[valid_mask]
std_errs = []
for ev, vv in zip(errors, var_vals):
    if vv is None or (isinstance(vv, float) and (vv <= 1e-12 or math.isnan(vv))):
        std_errs.append(float('nan'))
    else:
        std_errs.append(float(ev) / float(np.sqrt(vv)))
std_errs = np.asarray(std_errs)

q1_manual = float(np.nanmean(std_errs))
q2_manual = float(np.nanvar(std_errs))

tc5b_status, tc5b_err = check_scalar(
    "TC5b",
    "Q1 matches manual nanmean(standardized errors)",
    q1_manual,
    cv_result["q1"],
    1e-12
)
tc5c_status, tc5c_err = check_scalar(
    "TC5c",
    "Q2 matches manual nanvar(standardized errors)",
    q2_manual,
    cv_result["q2"],
    1e-12
)
print(f"TC5b [{tc5b_status}]: Q1 reported={cv_result['q1']:.8f}, manual={q1_manual:.8f}, |err|={tc5b_err:.2e}")
print(f"TC5c [{tc5c_status}]: Q2 reported={cv_result['q2']:.8f}, manual={q2_manual:.8f}, |err|={tc5c_err:.2e}")

# ── TC6: < 3 points -> NaN metrics returned gracefully ──────────────────────
x_tiny = np.array([0.0, 1.0])
y_tiny = np.array([0.0, 1.0])
h_tiny = np.array([1.0, 2.0])

cv_tiny = cross_validate(x_tiny, y_tiny, h_tiny, cfg_no_drift, vario)

tc6_status = check_true(
    "TC6",
    "With < 3 points, function returns NaN metrics gracefully",
    (
        math.isnan(cv_tiny["rmse"]) and
        math.isnan(cv_tiny["mae"]) and
        math.isnan(cv_tiny["q1"]) and
        math.isnan(cv_tiny["q2"])
    )
)
print(f"TC6 [{tc6_status}]: rmse={cv_tiny['rmse']}, mae={cv_tiny['mae']}, q1={cv_tiny['q1']}, q2={cv_tiny['q2']}")

# ── summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 80)
print(f"{'TC':<6} {'Description':<52} {'Tol':<12} {'Value':<14} {'Status'}")
print("-" * 80)
for (tc, desc, tol, val, status) in results:
    tol_str = f"{tol:.0e}" if tol is not None else "N/A"
    val_str = f"{val:.2e}" if val is not None else "N/A"
    print(f"{tc:<6} {desc:<52} {tol_str:<12} {val_str:<14} {status}")
print("=" * 80)

n_pass = sum(1 for r in results if r[4] == PASS)
n_fail = sum(1 for r in results if r[4] == FAIL)
print(f"\nSummary: {n_pass}/{len(results)} PASS, {n_fail} FAIL")

# ── SVG output ────────────────────────────────────────────────────────────────
import textwrap

# Layout: 3 plots in top row, full-width table in bottom row
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Task 3.10 - V&V: LOOCV Diagnostic Metrics",
             fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[1, 1.3],
                       hspace=0.50, wspace=0.35)

# Panel 1: Predicted vs Observed scatter
ax1 = fig.add_subplot(gs[0, 0])
valid_preds = preds[valid_mask]
valid_obs   = obs[valid_mask]
ax1.scatter(valid_obs, valid_preds, color="steelblue", edgecolors="k",
            linewidths=0.5, zorder=3, label="LOOCV predictions")
lo = min(valid_obs.min(), valid_preds.min()) - 1
hi = max(valid_obs.max(), valid_preds.max()) + 1
ax1.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="1:1 line")
ax1.set_xlabel("Observed")
ax1.set_ylabel("Predicted")
ax1.set_title("Predicted vs Observed (LOOCV)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2: Residuals histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(errors, bins=8, color="steelblue", edgecolor="k", alpha=0.8)
ax2.axvline(0, color="red", linestyle="--", linewidth=1.2)
ax2.set_xlabel("Residual (pred - obs)")
ax2.set_ylabel("Count")
ax2.set_title(f"Residual Distribution\nRMSE={cv_result['rmse']:.4f}  MAE={cv_result['mae']:.4f}")
ax2.grid(True, alpha=0.3)

# Panel 3: Standardized errors
ax3 = fig.add_subplot(gs[0, 2])
valid_std = std_errs[~np.isnan(std_errs)]
ax3.hist(valid_std, bins=8, color="darkorange", edgecolor="k", alpha=0.8)
ax3.axvline(0, color="red", linestyle="--", linewidth=1.2)
ax3.set_xlabel("Standardized Error (e / sqrt(var))")
ax3.set_ylabel("Count")
ax3.set_title(f"Standardized Errors\nQ1={cv_result['q1']:.4f}  Q2={cv_result['q2']:.4f}")
ax3.grid(True, alpha=0.3)

# Panel 4: Pass/Fail table — spans all 3 columns of bottom row
ax4 = fig.add_subplot(gs[1, :])
ax4.axis("off")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title(f"Test Results: {n_pass}/{len(results)} PASS",
              fontsize=11, fontweight="bold", pad=8)

# Column layout (axes-fraction)
col_x  = [0.01,  0.10,  0.88]   # left edges: TC, Description, Status
col_w  = [0.08,  0.77,  0.11]   # widths
wrap_chars = 90                   # generous wrap for full-width panel

# Header bar
hdr_y = 0.97
hdr_h = 0.09
ax4.add_patch(plt.Rectangle((0.0, hdr_y - hdr_h), 1.0, hdr_h,
                              facecolor="#aaaaaa", edgecolor="black",
                              linewidth=0.8, transform=ax4.transAxes, clip_on=False))
for label, x, w in zip(["TC", "Description", "Status"], col_x, col_w):
    ax4.text(x + w / 2, hdr_y - hdr_h / 2, label,
             ha="center", va="center", fontsize=9, fontweight="bold",
             transform=ax4.transAxes)
for xd in [col_x[1], col_x[2]]:
    ax4.plot([xd, xd], [hdr_y - hdr_h, hdr_y], color="black",
             linewidth=0.8, transform=ax4.transAxes)

# Row sizing
row_line_h = 0.065
row_pad    = 0.010

wrapped_descs = []
row_heights   = []
for (tc, desc, tol, val, status) in results:
    lines = textwrap.wrap(desc, width=wrap_chars)
    wrapped_descs.append(lines)
    row_heights.append(max(1, len(lines)) * row_line_h + row_pad)

# Draw rows
y = hdr_y - hdr_h
for (tc, desc, tol, val, status), lines, rh in zip(results, wrapped_descs, row_heights):
    colour = "#d4edda" if status == PASS else "#f8d7da"
    ax4.add_patch(plt.Rectangle((0.0, y - rh), 1.0, rh,
                                 facecolor=colour, edgecolor="black",
                                 linewidth=0.5, transform=ax4.transAxes, clip_on=False))
    mid_y = y - rh / 2
    ax4.text(col_x[0] + col_w[0] / 2, mid_y, tc,
             ha="center", va="center", fontsize=8.5, transform=ax4.transAxes)
    ax4.text(col_x[1] + 0.01, mid_y, "\n".join(lines),
             ha="left", va="center", fontsize=8, transform=ax4.transAxes,
             linespacing=1.3)
    sc = "#155724" if status == PASS else "#721c24"
    ax4.text(col_x[2] + col_w[2] / 2, mid_y, status,
             ha="center", va="center", fontsize=8.5, fontweight="bold",
             color=sc, transform=ax4.transAxes)
    for xd in [col_x[1], col_x[2]]:
        ax4.plot([xd, xd], [y - rh, y], color="black",
                 linewidth=0.5, transform=ax4.transAxes)
    y -= rh

svg_path = os.path.join(OUTPUT_DIR, "vv_loocv.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"\nSVG saved -> {svg_path}")
