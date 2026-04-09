"""
Task 3.3 -- V&V: Polynomial Drift Computation
==============================================
Verifies that compute_polynomial_drift() produces mathematically correct drift
columns, and that compute_resc() applies the safety floor correctly.

Outputs:
  - Console: summary table with PASS/FAIL per test case
  - docs/validation/output/vv_polynomial_drift.svg  (visualisation of drift columns)

Run from the project root:
    python docs/validation/vv_polynomial_drift.py
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

from drift import compute_resc, compute_polynomial_drift

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
TOL = 1e-14          # exact-match tolerance for drift values
RESC_TOL = 1e-14     # tolerance for resc comparisons

results = []   # list of (test_id, description, expected, actual, abs_err, status)


def check(test_id: str, description: str, expected: float, actual: float,
          tol: float = TOL) -> str:
    err = abs(actual - expected)
    status = PASS if err <= tol else FAIL
    results.append((test_id, description, expected, actual, err, status))
    return status


def check_array(test_id: str, description: str,
                expected: np.ndarray, actual: np.ndarray,
                tol: float = TOL) -> str:
    max_err = float(np.max(np.abs(actual - expected)))
    status = PASS if max_err <= tol else FAIL
    results.append((test_id, description,
                    f"array{list(expected)}", f"array{list(actual)}",
                    max_err, status))
    return status


# ---------------------------------------------------------------------------
# Test 1 — Linear X
# ---------------------------------------------------------------------------
# Given x=[0,10,20], y=[0,0,0], resc=0.5
# drift[:,0] should equal resc*x = [0, 5, 10]

x1 = np.array([0.0, 10.0, 20.0])
y1 = np.array([0.0,  0.0,  0.0])
resc1 = 0.5
cfg1 = {"drift_terms": {"linear_x": True}}
D1, names1 = compute_polynomial_drift(x1, y1, cfg1, resc1)

expected_1 = np.array([0.0, 5.0, 10.0])
check_array("TC-1", "Linear X: drift[:,0] = resc*x = [0, 5, 10]",
            expected_1, D1[:, 0])
check("TC-1b", "Linear X: only 1 column produced", 1, D1.shape[1])
check("TC-1c", "Linear X: term_names[0] == 'linear_x'",
      0, 0 if names1[0] == "linear_x" else 1)

# ---------------------------------------------------------------------------
# Test 2 — Linear Y
# ---------------------------------------------------------------------------
x2 = np.array([0.0, 0.0,  0.0])
y2 = np.array([0.0, 10.0, 20.0])
resc2 = 0.5
cfg2 = {"drift_terms": {"linear_y": True}}
D2, names2 = compute_polynomial_drift(x2, y2, cfg2, resc2)

expected_2 = np.array([0.0, 5.0, 10.0])
check_array("TC-2", "Linear Y: drift[:,0] = resc*y = [0, 5, 10]",
            expected_2, D2[:, 0])
check("TC-2b", "Linear Y: term_names[0] == 'linear_y'",
      0, 0 if names2[0] == "linear_y" else 1)

# ---------------------------------------------------------------------------
# Test 3 — Quadratic X
# ---------------------------------------------------------------------------
x3 = np.array([0.0, 10.0, 20.0])
y3 = np.array([0.0,  0.0,  0.0])
resc3 = 0.5
cfg3 = {"drift_terms": {"quadratic_x": True}}
D3, names3 = compute_polynomial_drift(x3, y3, cfg3, resc3)

# resc * x^2 = 0.5 * [0, 100, 400] = [0, 50, 200]
expected_3 = np.array([0.0, 50.0, 200.0])
check_array("TC-3", "Quadratic X: drift[:,0] = resc*x^2 = [0, 50, 200]",
            expected_3, D3[:, 0])
check("TC-3b", "Quadratic X: term_names[0] == 'quadratic_x'",
      0, 0 if names3[0] == "quadratic_x" else 1)

# ---------------------------------------------------------------------------
# Test 4 — Quadratic Y
# ---------------------------------------------------------------------------
x4 = np.array([0.0, 0.0,  0.0])
y4 = np.array([0.0, 10.0, 20.0])
resc4 = 0.5
cfg4 = {"drift_terms": {"quadratic_y": True}}
D4, names4 = compute_polynomial_drift(x4, y4, cfg4, resc4)

expected_4 = np.array([0.0, 50.0, 200.0])
check_array("TC-4", "Quadratic Y: drift[:,0] = resc*y^2 = [0, 50, 200]",
            expected_4, D4[:, 0])
check("TC-4b", "Quadratic Y: term_names[0] == 'quadratic_y'",
      0, 0 if names4[0] == "quadratic_y" else 1)

# ---------------------------------------------------------------------------
# Test 5 — Term ordering is always [linear_x, linear_y, quadratic_x, quadratic_y]
# regardless of the order keys appear in the config dict.
# ---------------------------------------------------------------------------
x5 = np.array([1.0, 2.0, 3.0])
y5 = np.array([4.0, 5.0, 6.0])
resc5 = 1.0

# Config with keys in reverse order
cfg5_reversed = {
    "drift_terms": {
        "quadratic_y": True,
        "quadratic_x": True,
        "linear_y": True,
        "linear_x": True,
    }
}
D5, names5 = compute_polynomial_drift(x5, y5, cfg5_reversed, resc5)

expected_order = ["linear_x", "linear_y", "quadratic_x", "quadratic_y"]
order_ok = (names5 == expected_order)
check("TC-5a", "Term ordering: names == [linear_x, linear_y, quadratic_x, quadratic_y]",
      0, 0 if order_ok else 1)
check("TC-5b", "Term ordering: 4 columns produced", 4, D5.shape[1])

# Verify column values match expected formulas
check_array("TC-5c", "Term ordering: col 0 = resc*x (linear_x)",
            resc5 * x5, D5[:, 0])
check_array("TC-5d", "Term ordering: col 1 = resc*y (linear_y)",
            resc5 * y5, D5[:, 1])
check_array("TC-5e", "Term ordering: col 2 = resc*x^2 (quadratic_x)",
            resc5 * x5**2, D5[:, 2])
check_array("TC-5f", "Term ordering: col 3 = resc*y^2 (quadratic_y)",
            resc5 * y5**2, D5[:, 3])

# ---------------------------------------------------------------------------
# Test 6a — compute_resc: normal case (no safety floor)
# x=[0,100], y=[0,100], sill=1.0, range=50
#   center = (50, 50)
#   radsqd = max((x-50)^2 + (y-50)^2) = max(5000, 5000) = 5000
#   BUT the plan says radsqd=10000 — let's check with the actual formula:
#   x-center = [-50, 50], y-center = [-50, 50]
#   (x-cx)^2 + (y-cy)^2 = [5000, 5000]  → radsqd = 5000
#   safe_radsqd = max(5000, 50^2=2500) = 5000
#   resc = sqrt(1/5000) ≈ 0.014142...
#
# NOTE: The plan's worked example uses radsqd=10000 which would require
# points at (0,0) and (100,100) measured from the origin, not the centroid.
# The actual implementation uses centroid-based radius. We test the actual
# implementation behaviour here and document the discrepancy.
# ---------------------------------------------------------------------------
x6a = np.array([0.0, 100.0])
y6a = np.array([0.0, 100.0])
sill6a = 1.0
range6a = 50.0

resc6a = compute_resc(sill6a, x6a, y6a, range6a)

# Centroid = (50, 50); max squared distance from centroid = 50^2+50^2 = 5000
# safe_radsqd = max(5000, 2500) = 5000
# resc = sqrt(1/5000)
expected_resc6a = math.sqrt(sill6a / 5000.0)
check("TC-6a", "compute_resc normal: resc = sqrt(sill/radsqd_centroid)",
      expected_resc6a, resc6a, tol=RESC_TOL)

# ---------------------------------------------------------------------------
# Test 6b — compute_resc: safety floor triggers
# x=[0,1], y=[0,1], sill=1.0, range=1000
#   center = (0.5, 0.5)
#   radsqd = max(0.5^2+0.5^2) = 0.5
#   safe_radsqd = max(0.5, 1000^2=1000000) = 1000000  ← floor triggers
#   resc = sqrt(1/1000000) = 0.001
# ---------------------------------------------------------------------------
x6b = np.array([0.0, 1.0])
y6b = np.array([0.0, 1.0])
sill6b = 1.0
range6b = 1000.0

resc6b = compute_resc(sill6b, x6b, y6b, range6b)

expected_resc6b = math.sqrt(sill6b / (range6b**2))   # = 0.001
check("TC-6b", "compute_resc safety floor: resc = sqrt(sill/range^2) = 0.001",
      expected_resc6b, resc6b, tol=RESC_TOL)

# Confirm floor actually triggered (radsqd < range^2)
x6b_center = np.mean(x6b)
y6b_center = np.mean(y6b)
radsqd_6b = float(np.max((x6b - x6b_center)**2 + (y6b - y6b_center)**2))
floor_triggered = radsqd_6b < range6b**2
check("TC-6b-floor", "compute_resc safety floor: floor triggered (radsqd < range^2)",
      0, 0 if floor_triggered else 1)

# ---------------------------------------------------------------------------
# Test 7 — Empty config produces zero-column matrix
# ---------------------------------------------------------------------------
x7 = np.array([1.0, 2.0, 3.0])
y7 = np.array([1.0, 2.0, 3.0])
cfg7 = {"drift_terms": {}}
D7, names7 = compute_polynomial_drift(x7, y7, cfg7, 1.0)
check("TC-7a", "Empty config: 0 columns", 0, D7.shape[1])
check("TC-7b", "Empty config: 0 term names", 0, len(names7))

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print(f"{'Test':<10} {'Description':<55} {'Expected':<18} {'Actual':<18} {'|Error|':<14} {'Status'}")
print("=" * 100)
all_pass = True
for tid, desc, exp, act, err, status in results:
    if status == FAIL:
        all_pass = False
    exp_str = f"{exp:.6g}" if isinstance(exp, float) else str(exp)
    act_str = f"{act:.6g}" if isinstance(act, float) else str(act)
    err_str = f"{err:.3e}"
    print(f"{tid:<10} {desc:<55} {exp_str:<18} {act_str:<18} {err_str:<14} {status}")

print("=" * 100)
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
print()

# ---------------------------------------------------------------------------
# SVG output — visualise each drift column for the all-terms case (TC-5)
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)
SVG_PATH = os.path.join(OUT_DIR, "vv_polynomial_drift.svg")

x_plot = np.linspace(0, 100, 200)
y_plot = np.linspace(0, 100, 200)
resc_plot = 0.01   # representative value

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    "Task 3.3 V&V — Polynomial Drift Columns\n"
    r"$\mathrm{resc}=0.01$, $x,y \in [0,100]$",
    fontsize=13,
)

plot_specs = [
    ("linear_x",    r"$D = \mathrm{resc} \cdot x$",    x_plot, resc_plot * x_plot,    "x", "steelblue"),
    ("linear_y",    r"$D = \mathrm{resc} \cdot y$",    y_plot, resc_plot * y_plot,    "y", "darkorange"),
    ("quadratic_x", r"$D = \mathrm{resc} \cdot x^2$",  x_plot, resc_plot * x_plot**2, "x", "seagreen"),
    ("quadratic_y", r"$D = \mathrm{resc} \cdot y^2$",  y_plot, resc_plot * y_plot**2, "y", "crimson"),
]

for ax, (name, formula, coord, values, xlabel, color) in zip(axes.flat, plot_specs):
    ax.plot(coord, values, color=color, linewidth=2)
    ax.set_title(f"{name}\n{formula}", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Drift value")
    ax.grid(True, alpha=0.3)

    # Overlay the three test points used in TC-1 / TC-3 style checks
    test_coords = np.array([0.0, 10.0, 20.0])
    if "x" in name:
        test_vals = resc_plot * (test_coords if "linear" in name else test_coords**2)
    else:
        test_vals = resc_plot * (test_coords if "linear" in name else test_coords**2)
    ax.scatter(test_coords, test_vals, color="black", zorder=5,
               label="Test points [0,10,20]")
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(SVG_PATH, format="svg")
PNG_PATH = os.path.join(OUT_DIR, "vv_polynomial_drift.png")
fig.savefig(PNG_PATH, format="png", dpi=150)
plt.close(fig)
print(f"SVG saved -> {SVG_PATH}")

# ---------------------------------------------------------------------------
# Second figure: resc safety floor illustration
# ---------------------------------------------------------------------------
SVG_PATH2 = os.path.join(OUT_DIR, "vv_polynomial_drift_resc.svg")

ranges = np.logspace(0, 4, 300)   # range from 1 to 10000
sill_demo = 1.0

# Case A: data extent >> range  (radsqd=5000, no floor)
radsqd_A = 5000.0
resc_A = np.sqrt(sill_demo / np.maximum(radsqd_A, ranges**2))

# Case B: data extent << range  (radsqd=0.5, floor always active)
radsqd_B = 0.5
resc_B = np.sqrt(sill_demo / np.maximum(radsqd_B, ranges**2))

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.loglog(ranges, np.sqrt(sill_demo / radsqd_A) * np.ones_like(ranges),
           "b--", alpha=0.4, label=r"No floor: $\sqrt{sill/radsqd}$")
ax2.loglog(ranges, resc_A, "b-", linewidth=2,
           label=r"radsqd=5000 (large extent)")
ax2.loglog(ranges, resc_B, "r-", linewidth=2,
           label=r"radsqd=0.5 (small extent, floor active)")
ax2.axvline(math.sqrt(radsqd_A), color="blue", linestyle=":", alpha=0.6,
            label=r"$\sqrt{radsqd_A}$")
ax2.set_xlabel("Variogram range")
ax2.set_ylabel("resc")
ax2.set_title(
    "compute_resc() — Safety Floor Behaviour\n"
    r"$\mathrm{resc} = \sqrt{sill \,/\, \max(radsqd,\, range^2)}$",
    fontsize=12,
)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)
plt.tight_layout()
fig2.savefig(SVG_PATH2, format="svg")
PNG_PATH2 = os.path.join(OUT_DIR, "vv_polynomial_drift_resc.png")
fig2.savefig(PNG_PATH2, format="png", dpi=150)
plt.close(fig2)
print(f"SVG saved -> {SVG_PATH2}")

# Exit with non-zero code if any test failed
sys.exit(0 if all_pass else 1)
