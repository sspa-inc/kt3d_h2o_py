"""
Task 3.6 — V&V: AEM Drift Matrix Scaling Consistency
=====================================================
Objective: Verify that AEM drift scaling factors from training are correctly
reused during prediction via compute_linesink_drift_matrix().

Test cases:
  TC1: Scaling factors from training are returned correctly (adaptive method)
  TC2: Prediction with input_scaling_factors reuses training factors exactly
  TC3: Prediction WITHOUT input_scaling_factors produces DIFFERENT factors
  TC4: Fixed rescaling method produces sill / 0.0001 regardless of data
  TC5: SVG output — potential field and scaling factor comparison plot

Output: docs/validation/output/vv_aem_scaling_consistency.svg
"""

import sys
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AEM_drift import compute_linesink_drift_matrix

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []   # (tc_label, description, expected, actual, error, status)

def check(label, description, expected, actual, tol, fmt=".6e"):
    err = abs(actual - expected)
    status = PASS if err <= tol else FAIL
    results.append((label, description, expected, actual, err, status))
    return status

def check_exact_dict(label, description, dict_a, dict_b, tol=1e-15):
    """Compare two dicts of floats key-by-key."""
    if set(dict_a.keys()) != set(dict_b.keys()):
        results.append((label, description, "same keys", "different keys", float("nan"), FAIL))
        return FAIL
    worst = 0.0
    for k in dict_a:
        worst = max(worst, abs(dict_a[k] - dict_b[k]))
    status = PASS if worst <= tol else FAIL
    results.append((label, description, 0.0, worst, worst, status))
    return status

# ── build synthetic linesink GeoDataFrame ────────────────────────────────────
# Group A: horizontal segment at y=50
# Group B: vertical segment at x=50
seg_A = LineString([(0, 50), (100, 50)])
seg_B = LineString([(50, 0), (50, 100)])

gdf = gpd.GeoDataFrame(
    {
        "group": ["A", "B"],
        "resistance": [1.0, 1.0],
    },
    geometry=[seg_A, seg_B],
    crs="EPSG:28992",
)

GROUP_COL = "group"
STRENGTH_COL = "resistance"
SILL = 2.5

# No anisotropy transform needed for this test
TRANSFORM_PARAMS = None

# ── training points (N=10, seed=42) ──────────────────────────────────────────
rng = np.random.default_rng(42)
x_train = rng.uniform(10, 90, 10)
y_train = rng.uniform(10, 90, 10)

# ── prediction points (N=20, seed=99) ────────────────────────────────────────
rng2 = np.random.default_rng(99)
x_pred = rng2.uniform(10, 90, 20)
y_pred = rng2.uniform(10, 90, 20)

# ════════════════════════════════════════════════════════════════════════════
# TC1 — Training returns scaling_factors dict with correct keys
# ════════════════════════════════════════════════════════════════════════════
drift_train, names_train, sf_train = compute_linesink_drift_matrix(
    x_train, y_train, gdf, GROUP_COL,
    transform_params=TRANSFORM_PARAMS,
    sill=SILL,
    strength_col=STRENGTH_COL,
    rescaling_method="adaptive",
    apply_anisotropy=False,
    input_scaling_factors=None,
)

tc1_keys_ok = set(sf_train.keys()) == {"A", "B"}
tc1_shape_ok = drift_train.shape == (10, 2)
tc1_status = PASS if (tc1_keys_ok and tc1_shape_ok) else FAIL
results.append((
    "TC1",
    "Training returns sf dict with keys {A,B} and matrix shape (10,2)",
    "keys={A,B}, shape=(10,2)",
    f"keys={set(sf_train.keys())}, shape={drift_train.shape}",
    0.0 if tc1_status == PASS else 1.0,
    tc1_status,
))

# ════════════════════════════════════════════════════════════════════════════
# TC2 — Prediction WITH input_scaling_factors reuses training factors exactly
# ════════════════════════════════════════════════════════════════════════════
drift_pred_with, names_pred_with, sf_pred_with = compute_linesink_drift_matrix(
    x_pred, y_pred, gdf, GROUP_COL,
    transform_params=TRANSFORM_PARAMS,
    sill=SILL,
    strength_col=STRENGTH_COL,
    rescaling_method="adaptive",
    apply_anisotropy=False,
    input_scaling_factors=sf_train,
)

check_exact_dict(
    "TC2",
    "Prediction scaling factors match training factors exactly (tol=1e-15)",
    sf_train, sf_pred_with, tol=1e-15,
)

# ════════════════════════════════════════════════════════════════════════════
# TC3 — Prediction WITHOUT input_scaling_factors produces DIFFERENT factors
# ════════════════════════════════════════════════════════════════════════════
drift_pred_free, names_pred_free, sf_pred_free = compute_linesink_drift_matrix(
    x_pred, y_pred, gdf, GROUP_COL,
    transform_params=TRANSFORM_PARAMS,
    sill=SILL,
    strength_col=STRENGTH_COL,
    rescaling_method="adaptive",
    apply_anisotropy=False,
    input_scaling_factors=None,
)

# At least one factor should differ between training and free-prediction
any_differ = any(abs(sf_pred_free[k] - sf_train[k]) > 1e-15 for k in sf_train)
tc3_status = PASS if any_differ else FAIL
results.append((
    "TC3",
    "Free prediction (no input_sf) produces different scaling factors",
    "at least one factor differs",
    "differs" if any_differ else "identical",
    0.0 if tc3_status == PASS else 1.0,
    tc3_status,
))

# ════════════════════════════════════════════════════════════════════════════
# TC4 — Fixed rescaling method: factor == sill / 0.0001 for every group
# ════════════════════════════════════════════════════════════════════════════
expected_fixed = SILL / 0.0001

drift_fixed, names_fixed, sf_fixed = compute_linesink_drift_matrix(
    x_train, y_train, gdf, GROUP_COL,
    transform_params=TRANSFORM_PARAMS,
    sill=SILL,
    strength_col=STRENGTH_COL,
    rescaling_method="fixed",
    apply_anisotropy=False,
    input_scaling_factors=None,
)

for grp in ["A", "B"]:
    check(
        "TC4",
        f"Fixed method: sf['{grp}'] == sill/0.0001 = {expected_fixed:.6e}",
        expected_fixed,
        sf_fixed[grp],
        tol=1e-10,
    )

# ════════════════════════════════════════════════════════════════════════════
# TC5 — Drift column values: prediction with reused factors == phi * sf_train
# ════════════════════════════════════════════════════════════════════════════
# Manually compute expected drift column for group A at prediction points
from AEM_drift import compute_linesink_potential

phi_A_pred = compute_linesink_potential(
    x_pred, y_pred,
    seg_A.coords[0][0], seg_A.coords[0][1],
    seg_A.coords[1][0], seg_A.coords[1][1],
    strength=1.0,
)
expected_col_A = phi_A_pred * sf_train["A"]
actual_col_A = drift_pred_with[:, names_pred_with.index("A")]
max_err_A = np.max(np.abs(actual_col_A - expected_col_A))

check(
    "TC5",
    "Drift column A at pred points == phi_A * sf_train['A'] (tol=1e-12)",
    0.0, max_err_A, tol=1e-12,
)

phi_B_pred = compute_linesink_potential(
    x_pred, y_pred,
    seg_B.coords[0][0], seg_B.coords[0][1],
    seg_B.coords[1][0], seg_B.coords[1][1],
    strength=1.0,
)
expected_col_B = phi_B_pred * sf_train["B"]
actual_col_B = drift_pred_with[:, names_pred_with.index("B")]
max_err_B = np.max(np.abs(actual_col_B - expected_col_B))

check(
    "TC5",
    "Drift column B at pred points == phi_B * sf_train['B'] (tol=1e-12)",
    0.0, max_err_B, tol=1e-12,
)

# ════════════════════════════════════════════════════════════════════════════
# Print summary table
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("Task 3.6 - V&V: AEM Drift Matrix Scaling Consistency")
print("=" * 90)
header = f"{'TC':<6} {'Description':<55} {'Error':>12}  {'Status'}"
print(header)
print("-" * 90)
for (tc, desc, exp, act, err, status) in results:
    err_str = f"{err:.3e}" if isinstance(err, float) else str(err)
    print(f"{tc:<6} {desc:<55} {err_str:>12}  {status}")

n_pass = sum(1 for r in results if r[-1] == PASS)
n_fail = sum(1 for r in results if r[-1] == FAIL)
print("-" * 90)
print(f"TOTAL: {n_pass} PASS, {n_fail} FAIL")
print("=" * 90)

# ════════════════════════════════════════════════════════════════════════════
# SVG output — 4-panel figure
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Task 3.6 — AEM Drift Matrix Scaling Consistency", fontsize=13, fontweight="bold")

# Panel 1: Linesink geometry + training/prediction points
ax = axes[0, 0]
ax.set_title("Linesink Geometry & Sample Points", fontsize=10)
ax.plot([0, 100], [50, 50], "b-", lw=2, label="Group A (horizontal)")
ax.plot([50, 50], [0, 100], "r-", lw=2, label="Group B (vertical)")
ax.scatter(x_train, y_train, c="steelblue", s=50, zorder=5, label=f"Training (N={len(x_train)})")
ax.scatter(x_pred, y_pred, c="orange", marker="^", s=50, zorder=5, label=f"Prediction (N={len(x_pred)})")
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(fontsize=8)
ax.set_aspect("equal")

# Panel 2: Scaling factors comparison (training vs free-prediction vs fixed)
ax = axes[0, 1]
ax.set_title("Scaling Factors by Group", fontsize=10)
groups = ["A", "B"]
x_pos = np.arange(len(groups))
width = 0.25
bars1 = ax.bar(x_pos - width, [sf_train[g] for g in groups], width, label="Training (adaptive)", color="steelblue")
bars2 = ax.bar(x_pos,         [sf_pred_with[g] for g in groups], width, label="Pred w/ input_sf (reused)", color="limegreen")
bars3 = ax.bar(x_pos + width, [sf_pred_free[g] for g in groups], width, label="Pred w/o input_sf (free)", color="tomato")
ax.set_xticks(x_pos)
ax.set_xticklabels(groups)
ax.set_ylabel("Scaling Factor")
ax.legend(fontsize=8)
ax.set_xlabel("Linesink Group")

# Panel 3: Drift column A — training vs prediction (reused vs free)
ax = axes[1, 0]
ax.set_title("Drift Column A: Training vs Prediction", fontsize=10)
# Sort training by x for a cleaner line
idx_tr = np.argsort(x_train)
ax.plot(x_train[idx_tr], drift_train[idx_tr, names_train.index("A")],
        "o-", color="steelblue", ms=5, label="Training drift A")
idx_pr = np.argsort(x_pred)
ax.plot(x_pred[idx_pr], drift_pred_with[idx_pr, names_pred_with.index("A")],
        "^--", color="limegreen", ms=5, label="Pred (reused sf) drift A")
ax.plot(x_pred[idx_pr], drift_pred_free[idx_pr, names_pred_free.index("A")],
        "s:", color="tomato", ms=5, label="Pred (free sf) drift A")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Drift value")
ax.legend(fontsize=8)

# Panel 4: Pass/Fail summary table
ax = axes[1, 1]
ax.axis("off")
ax.set_title("Test Results Summary", fontsize=10)
col_labels = ["TC", "Description (short)", "Error", "Status"]
table_data = []
for (tc, desc, exp, act, err, status) in results:
    short_desc = desc[:40] + "…" if len(desc) > 40 else desc
    err_str = f"{err:.2e}" if isinstance(err, float) else str(err)
    table_data.append([tc, short_desc, err_str, status])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.auto_set_column_width([0, 1, 2, 3])

# Colour rows by status
for row_idx, (_, _, _, _, _, status) in enumerate(results):
    colour = "#d4edda" if status == PASS else "#f8d7da"
    for col_idx in range(4):
        tbl[(row_idx + 1, col_idx)].set_facecolor(colour)

# Summary line
summary_text = f"TOTAL: {n_pass} PASS, {n_fail} FAIL"
summary_colour = "green" if n_fail == 0 else "red"
ax.text(0.5, 0.02, summary_text, ha="center", va="bottom",
        transform=ax.transAxes, fontsize=10, fontweight="bold", color=summary_colour)

plt.tight_layout()
svg_path = os.path.join(OUTPUT_DIR, "vv_aem_scaling_consistency.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
png_path_sc = os.path.join(OUTPUT_DIR, "vv_aem_scaling_consistency.png")
fig.savefig(png_path_sc, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSVG saved -> {svg_path}")

# Exit with non-zero code if any test failed
if n_fail > 0:
    sys.exit(1)
