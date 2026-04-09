"""
Task 3.1 -- V&V: Variogram Model Equations
===========================================
Verifies that each variogram model in variogram.py produces correct semivariance
values against hand-calculated analytical values.

Outputs:
  - Console: summary table with PASS/FAIL per test case
  - docs/validation/output/vv_variogram_models.svg  (comparison plot)

Run from the project root:
    python docs/validation/vv_variogram_models.py
"""

import sys
import os
import math

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from variogram import variogram

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SILL   = 10.0
RANGE  = 100.0
NUGGET = 1.0
PSILL  = SILL - NUGGET   # = 9.0
TOL    = 1e-10

results = []   # list of (model, label, h, expected, actual, error, pass_)


def _make_vario(model_type):
    cfg = {
        "variogram": {
            "model": model_type,
            "sill": SILL,
            "range": RANGE,
            "nugget": NUGGET,
        }
    }
    return variogram(config=cfg)


def check(model_name, label, h, expected, actual, tol=TOL):
    err = abs(actual - expected)
    pass_ = err < tol
    results.append((model_name, label, h, expected, actual, err, pass_))
    return pass_


# ---------------------------------------------------------------------------
# Analytical reference formulas (hand-calculated)
# ---------------------------------------------------------------------------

def spherical_ref(h, sill, range_, nugget):
    psill = sill - nugget
    if h <= range_:
        return nugget + psill * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3)
    return sill


def exponential_ref(h, sill, range_, nugget):
    psill = sill - nugget
    return nugget + psill * (1 - math.exp(-h / (range_ / 3.0)))


def gaussian_ref(h, sill, range_, nugget):
    psill = sill - nugget
    return nugget + psill * (1 - math.exp(-(h / (range_ / math.sqrt(3.0))) ** 2))


def linear_ref(h, sill, range_, nugget):
    psill = sill - nugget
    if h <= range_:
        return nugget + (psill / range_) * h
    return sill


REFS = {
    "spherical":   spherical_ref,
    "exponential": exponential_ref,
    "gaussian":    gaussian_ref,
    "linear":      linear_ref,
}

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

for model_name, ref_fn in REFS.items():
    v = _make_vario(model_name)

    # --- h = 0 ---
    # All four models satisfy gamma(0) = nugget because the psill term vanishes.
    h0_expected = NUGGET
    h0_actual   = v.calculate_variogram(0.0)
    check(model_name, "h=0 -> nugget", 0.0, h0_expected, h0_actual)

    # --- h = range ---
    # Spherical and linear reach sill exactly at h=range.
    # Exponential and Gaussian approach sill asymptotically.
    h_range_expected = ref_fn(RANGE, SILL, RANGE, NUGGET)
    h_range_actual   = v.calculate_variogram(RANGE)
    if model_name in ("spherical", "linear"):
        check(model_name, "h=range -> sill (exact)", RANGE, h_range_expected, h_range_actual)
    else:
        check(model_name, "h=range (asymptotic formula)", RANGE, h_range_expected, h_range_actual)

    # --- h = 2*range (bounded models only) ---
    if model_name in ("spherical", "linear"):
        h2r_expected = SILL
        h2r_actual   = v.calculate_variogram(2 * RANGE)
        check(model_name, "h=2*range -> sill (bounded)", 2 * RANGE, h2r_expected, h2r_actual)

    # --- h = range/2 (hand-calculated) ---
    h_half = RANGE / 2.0
    h_half_expected = ref_fn(h_half, SILL, RANGE, NUGGET)
    h_half_actual   = v.calculate_variogram(h_half)
    check(model_name, "h=range/2 (analytical)", h_half, h_half_expected, h_half_actual)

# ---------------------------------------------------------------------------
# Validation: nugget >= sill raises ValueError
# ---------------------------------------------------------------------------
nugget_error_raised = False
try:
    bad_cfg = {"variogram": {"model": "spherical", "sill": 1.0, "nugget": 1.5, "range": 100.0}}
    variogram(config=bad_cfg)
except ValueError:
    nugget_error_raised = True

results.append(("(validation)", "nugget>=sill raises ValueError", None,
                True, nugget_error_raised, 0.0, nugget_error_raised))

# ---------------------------------------------------------------------------
# Validation: range <= 0 raises ValueError
# ---------------------------------------------------------------------------
range_error_raised = False
try:
    bad_cfg = {"variogram": {"model": "spherical", "sill": 1.0, "nugget": 0.0, "range": -10.0}}
    variogram(config=bad_cfg)
except ValueError:
    range_error_raised = True

results.append(("(validation)", "range<0 raises ValueError", None,
                True, range_error_raised, 0.0, range_error_raised))

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
COL_W = 120
header = (
    f"{'Model':<14} {'Test Case':<38} {'h':>8}  "
    f"{'Expected':>12}  {'Actual':>12}  {'|Error|':>12}  Result"
)
print()
print("=" * COL_W)
print("Task 3.1 -- V&V: Variogram Model Equations")
print(f"  sill={SILL}, range={RANGE}, nugget={NUGGET}, psill={PSILL}")
print(f"  Tolerance: {TOL}")
print("=" * COL_W)
print(header)
print("-" * COL_W)

all_pass = True
for (model_name, label, h, expected, actual, err, pass_) in results:
    h_str = f"{h:8.2f}" if h is not None else f"{'N/A':>8}"
    if isinstance(expected, bool):
        exp_str = f"{'True':>12}"
        act_str = f"{str(actual):>12}"
        err_str = f"{'---':>12}"
    else:
        exp_str = f"{expected:12.8f}"
        act_str = f"{actual:12.8f}"
        err_str = f"{err:12.2e}"
    status = "PASS" if pass_ else "FAIL"
    if not pass_:
        all_pass = False
    print(f"{model_name:<14} {label:<38} {h_str}  {exp_str}  {act_str}  {err_str}  {status}")

print("-" * COL_W)
overall = "ALL TESTS PASSED" if all_pass else "*** SOME TESTS FAILED ***"
print(f"\n  Overall: {overall}\n")

# ---------------------------------------------------------------------------
# SVG comparison plot -- all four models, sill=10, range=100, nugget=1
# ---------------------------------------------------------------------------
h_vals = np.linspace(0, 250, 500)

fig, ax = plt.subplots(figsize=(8, 5))

colors = {
    "spherical":   "#1f77b4",
    "exponential": "#ff7f0e",
    "gaussian":    "#2ca02c",
    "linear":      "#d62728",
}

for model_name in ("spherical", "exponential", "gaussian", "linear"):
    v = _make_vario(model_name)
    gamma = [v.calculate_variogram(h) for h in h_vals]
    ax.plot(h_vals, gamma, label=model_name.capitalize(),
            color=colors[model_name], linewidth=2)

ax.axhline(SILL,   color="black", linestyle="--", linewidth=1, label=f"Sill = {SILL}")
ax.axhline(NUGGET, color="gray",  linestyle=":",  linewidth=1, label=f"Nugget = {NUGGET}")
ax.axvline(RANGE,  color="black", linestyle="-.", linewidth=1, label=f"Range = {RANGE}")

ax.set_xlabel("Lag distance h")
ax.set_ylabel("Semivariance gamma(h)")
ax.set_title(f"Variogram Models  (sill={SILL}, range={RANGE}, nugget={NUGGET})")
ax.legend(loc="lower right")
ax.set_xlim(0, 250)
ax.set_ylim(0, SILL * 1.15)
ax.grid(True, alpha=0.3)

out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)
svg_path = os.path.join(out_dir, "vv_variogram_models.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
png_path = os.path.join(out_dir, "vv_variogram_models.png")
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Plot saved -> {svg_path}\n")

# ---------------------------------------------------------------------------
# Exit with non-zero code on failure so CI can detect it
# ---------------------------------------------------------------------------
if not all_pass:
    sys.exit(1)
