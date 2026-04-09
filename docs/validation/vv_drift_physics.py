"""
Task 3.4 -- V&V: Drift Physics Verification System
===================================================
Verifies that verify_drift_physics() correctly identifies valid and invalid
drift columns, and returns the correct PASS / FAIL / SKIP status for each case.

Outputs:
  - Console: summary table with PASS/FAIL per test case
  - docs/validation/output/vv_drift_physics.svg  (visualisation of each test case)

Run from the project root:
    python docs/validation/vv_drift_physics.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drift import compute_resc, verify_drift_physics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"

results = []   # list of (test_id, description, expected, actual, status)


def check_status(test_id, description, expected, actual):
    status = PASS if actual == expected else FAIL
    results.append((test_id, description, expected, actual, status))
    return status


def print_summary():
    header = "%-8s %-55s %-10s %-10s %s" % ("Test", "Description", "Expected", "Actual", "Status")
    print()
    print(header)
    print("-" * len(header))
    all_pass = True
    for test_id, desc, expected, actual, status in results:
        print("%-8s %-55s %-10s %-10s %s" % (test_id, desc, expected, actual, status))
        if status != PASS:
            all_pass = False
    print()
    total = len(results)
    passed = sum(1 for r in results if r[4] == PASS)
    print("Result: %d/%d tests passed" % (passed, total))
    if all_pass:
        print("Overall: PASS")
    else:
        print("Overall: FAIL")
    return all_pass


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)

# 30 points spread over [0, 100] x [0, 100]
N = 30
x = rng.uniform(0, 100, N)
y = rng.uniform(0, 100, N)

# Compute resc using the same formula as the production code
# sill=1.0, range=50 -> radsqd = max squared distance from centroid
sill = 1.0
variogram_range = 50.0
resc = compute_resc(sill, x, y, variogram_range)

# ---------------------------------------------------------------------------
# Test 1 -- Valid linear_x: drift_col = resc * x  ->  expect PASS
# ---------------------------------------------------------------------------

drift_col_1 = resc * x
drift_matrix_1 = drift_col_1.reshape(-1, 1)
term_names_1 = ["linear_x"]

result_1 = verify_drift_physics(drift_matrix_1, term_names_1, x, y, resc)
check_status("TC-1", "Valid linear_x (drift = resc*x)", "PASS", result_1["linear_x"])

# ---------------------------------------------------------------------------
# Test 2 -- Valid quadratic_y: drift_col = resc * y^2  ->  expect PASS
# ---------------------------------------------------------------------------

drift_col_2 = resc * y**2
drift_matrix_2 = drift_col_2.reshape(-1, 1)
term_names_2 = ["quadratic_y"]

result_2 = verify_drift_physics(drift_matrix_2, term_names_2, x, y, resc)
check_status("TC-2", "Valid quadratic_y (drift = resc*y^2)", "PASS", result_2["quadratic_y"])

# ---------------------------------------------------------------------------
# Test 3 -- Corrupted linear_x: drift = resc*x + noise(scale=10%*resc*x)
#           R^2 will drop below 0.999  ->  expect FAIL
# ---------------------------------------------------------------------------

noise_scale = resc * np.abs(x).mean() * 0.10   # 10% of typical magnitude
noise = rng.normal(0, noise_scale, N)
drift_col_3 = resc * x + noise
drift_matrix_3 = drift_col_3.reshape(-1, 1)
term_names_3 = ["linear_x"]

result_3 = verify_drift_physics(drift_matrix_3, term_names_3, x, y, resc)
check_status("TC-3", "Corrupted linear_x (10% noise -> R^2<0.999)", "FAIL", result_3["linear_x"])

# ---------------------------------------------------------------------------
# Test 4 -- Wrong scaling: drift = 2*resc*x  ->  slope error = 100%  ->  expect FAIL
# ---------------------------------------------------------------------------

drift_col_4 = 2.0 * resc * x
drift_matrix_4 = drift_col_4.reshape(-1, 1)
term_names_4 = ["linear_x"]

result_4 = verify_drift_physics(drift_matrix_4, term_names_4, x, y, resc)
check_status("TC-4", "Wrong scaling (2*resc*x -> slope error 100%)", "FAIL", result_4["linear_x"])

# ---------------------------------------------------------------------------
# Test 5 -- AEM term (no _x or _y in name)  ->  expect SKIP
# ---------------------------------------------------------------------------

drift_col_5 = rng.uniform(0, 1, N)
drift_matrix_5 = drift_col_5.reshape(-1, 1)
term_names_5 = ["river_group_A"]   # AEM term -- no _x or _y suffix

result_5 = verify_drift_physics(drift_matrix_5, term_names_5, x, y, resc)
check_status("TC-5", "AEM term (no _x/_y) -> SKIP", "SKIP", result_5["river_group_A"])

# ---------------------------------------------------------------------------
# SVG output
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SVG_PATH = os.path.join(OUTPUT_DIR, "vv_drift_physics.svg")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("V&V 3.4 -- Drift Physics Verification System", fontsize=14, fontweight="bold")

x_line = np.linspace(x.min(), x.max(), 100)
y_sorted = np.sort(y)

# ---- Panel 1: TC-1 valid linear_x ----
ax = axes[0, 0]
ax.scatter(x, drift_col_1, s=20, color="steelblue", label="drift = resc*x")
ax.plot(x_line, resc * x_line, "r--", lw=1.5, label="resc*x (resc=%.4f)" % resc)
tc1_color = "green" if result_1["linear_x"] == "PASS" else "red"
ax.set_title("TC-1: Valid linear_x -> %s" % result_1["linear_x"], color=tc1_color)
ax.set_xlabel("x")
ax.set_ylabel("drift value")
ax.legend(fontsize=7)

# ---- Panel 2: TC-2 valid quadratic_y ----
ax = axes[0, 1]
ax.scatter(y, drift_col_2, s=20, color="steelblue", label="drift = resc*y^2")
ax.plot(y_sorted, resc * y_sorted**2, "r--", lw=1.5, label="resc*y^2")
tc2_color = "green" if result_2["quadratic_y"] == "PASS" else "red"
ax.set_title("TC-2: Valid quadratic_y -> %s" % result_2["quadratic_y"], color=tc2_color)
ax.set_xlabel("y")
ax.set_ylabel("drift value")
ax.legend(fontsize=7)

# ---- Panel 3: TC-3 corrupted linear_x ----
ax = axes[0, 2]
ax.scatter(x, drift_col_3, s=20, color="orange", label="resc*x + noise")
ax.plot(x_line, resc * x_line, "r--", lw=1.5, label="ideal resc*x")
tc3_color = "green" if result_3["linear_x"] == "PASS" else "red"
ax.set_title("TC-3: Corrupted linear_x -> %s" % result_3["linear_x"], color=tc3_color)
ax.set_xlabel("x")
ax.set_ylabel("drift value")
ax.legend(fontsize=7)

# ---- Panel 4: TC-4 wrong scaling ----
ax = axes[1, 0]
ax.scatter(x, drift_col_4, s=20, color="orange", label="2*resc*x")
ax.plot(x_line, resc * x_line, "r--", lw=1.5, label="ideal resc*x")
ax.plot(x_line, 2 * resc * x_line, "b-", lw=1.5, label="2*resc*x (actual)")
tc4_color = "green" if result_4["linear_x"] == "PASS" else "red"
ax.set_title("TC-4: Wrong scaling (2x) -> %s" % result_4["linear_x"], color=tc4_color)
ax.set_xlabel("x")
ax.set_ylabel("drift value")
ax.legend(fontsize=7)

# ---- Panel 5: TC-5 AEM term (SKIP) ----
ax = axes[1, 1]
ax.scatter(x, drift_col_5, s=20, color="gray", label="AEM potential (random)")
ax.set_title("TC-5: AEM term 'river_group_A' -> %s" % result_5["river_group_A"], color="purple")
ax.set_xlabel("x")
ax.set_ylabel("drift value")
ax.legend(fontsize=7)
ax.text(0.5, 0.5, "SKIP\n(no _x or _y in name)", transform=ax.transAxes,
        ha="center", va="center", fontsize=12, color="purple",
        bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.8))

# ---- Panel 6: Summary table ----
ax = axes[1, 2]
ax.axis("off")
table_data = [
    ["TC", "Description", "Exp.", "Got", "Status"],
    ["TC-1", "linear_x = resc*x", "PASS", result_1["linear_x"], result_1["linear_x"]],
    ["TC-2", "quadratic_y = resc*y^2", "PASS", result_2["quadratic_y"], result_2["quadratic_y"]],
    ["TC-3", "linear_x + 10% noise", "FAIL", result_3["linear_x"], result_3["linear_x"]],
    ["TC-4", "linear_x scaled x2", "FAIL", result_4["linear_x"], result_4["linear_x"]],
    ["TC-5", "AEM (no _x/_y)", "SKIP", result_5["river_group_A"], result_5["river_group_A"]],
]
colors = [["#dddddd"] * 5]
for row in table_data[1:]:
    status = row[4]
    if status == "PASS":
        c = "#c8f7c5"
    elif status == "FAIL":
        c = "#f7c5c5"
    else:
        c = "#e8e8f7"
    colors.append([c] * 5)

tbl = ax.table(cellText=table_data, cellLoc="center", loc="center",
               cellColours=colors,
               colWidths=[0.15, 0.40, 0.15, 0.15, 0.15])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.8)
ax.set_title("Summary", fontweight="bold")

plt.tight_layout()
fig.savefig(SVG_PATH, format="svg", bbox_inches="tight")
plt.close(fig)
print("\nSVG saved -> %s" % SVG_PATH)

# ---------------------------------------------------------------------------
# Print console summary
# ---------------------------------------------------------------------------

all_pass = print_summary()
sys.exit(0 if all_pass else 1)
