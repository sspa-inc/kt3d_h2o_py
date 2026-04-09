# -*- coding: utf-8 -*-
"""
Task 3.2 -- V&V: Coordinate Transformation Roundtrip
=====================================================

Objective:
    Verify that apply_transform() followed by invert_transform_coords() recovers
    original coordinates exactly (max absolute error < 1e-12).

Additional checks:
    - angle=0 deg (North), ratio=0.5: major axis (N/S) maps to X, minor axis (E/W) scaled by 2x on Y
    - angle=90 deg (East), ratio=1.0: no rotation (identity), coordinates unchanged

Orientation note (azimuth convention, CW from North, 0°=North):
    angle_major=0  -> major axis along North (+Y).
                      Internally theta=90° (arithmetic). R rotates 90° CCW.
                      Forward: [x_c, y_c] @ R = [y_c, -x_c].
    angle_major=90 -> major axis along East (+X).
                      Internally theta=0° (arithmetic). R = identity.
                      Forward: coordinates unchanged.

Outputs:
    docs/validation/output/vv_transform_roundtrip.svg
"""

import sys
import os
import math
import numpy as np

# ---------------------------------------------------------------------------
# Path setup -- allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transform import get_transform_params, apply_transform, invert_transform_coords

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(_HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROUNDTRIP_TOL = 1e-12

def roundtrip_error(x, y, angle_deg, ratio):
    """Forward-transform then invert; return max absolute error."""
    params = get_transform_params(x, y, angle_deg, ratio)
    xp, yp = apply_transform(x, y, params)
    xr, yr = invert_transform_coords(xp, yp, params)
    err_x = np.max(np.abs(xr - x))
    err_y = np.max(np.abs(yr - y))
    return max(err_x, err_y)


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = "  [%s] %s" % (status, label)
    if detail:
        msg += "  (%s)" % detail
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
results = []   # list of (label, passed)

print("=" * 65)
print("Task 3.2 -- V&V: Coordinate Transformation Roundtrip")
print("  Convention: azimuth (CW from North, 0 deg=North)")
print("=" * 65)

# -- Test 1: angle=90 deg (East) = identity (arithmetic 0°, no rotation) ----
print("\nTest 1: Random N=50, angle=90 deg (East), ratio=1.0 (identity)")
rng = np.random.default_rng(42)
x1 = rng.uniform(0, 1000, 50)
y1 = rng.uniform(0, 1000, 50)
err1 = roundtrip_error(x1, y1, 90.0, 1.0)
p1 = check("Roundtrip error < 1e-12", err1 < ROUNDTRIP_TOL, "max_err=%.3e" % err1)
results.append(("Test 1 (identity, az=90)", p1))

# -- Test 2: angle=45 deg (NE), ratio=0.5 -----------------------------------
print("\nTest 2: Random N=50, angle=45 deg (NE), ratio=0.5")
x2 = rng.uniform(0, 1000, 50)
y2 = rng.uniform(0, 1000, 50)
err2 = roundtrip_error(x2, y2, 45.0, 0.5)
p2 = check("Roundtrip error < 1e-12", err2 < ROUNDTRIP_TOL, "max_err=%.3e" % err2)
results.append(("Test 2 (45 deg, 0.5)", p2))

# -- Test 3: angle=90 deg (East), ratio=0.3 ---------------------------------
print("\nTest 3: Random N=50, angle=90 deg (East), ratio=0.3")
x3 = rng.uniform(0, 1000, 50)
y3 = rng.uniform(0, 1000, 50)
err3 = roundtrip_error(x3, y3, 90.0, 0.3)
p3 = check("Roundtrip error < 1e-12", err3 < ROUNDTRIP_TOL, "max_err=%.3e" % err3)
results.append(("Test 3 (90 deg, 0.3)", p3))

# -- Test 4: angle=0 deg (North), ratio=0.5 (rotation + scaling) ------------
# Azimuth 0 deg -> arithmetic 90 deg -> R rotates 90 deg CCW internally
print("\nTest 4: Random N=50, angle=0 deg (North), ratio=0.5 (rotation + scaling)")
x4 = rng.uniform(0, 1000, 50)
y4 = rng.uniform(0, 1000, 50)
err4 = roundtrip_error(x4, y4, 0.0, 0.5)
p4 = check("Roundtrip error < 1e-12", err4 < ROUNDTRIP_TOL, "max_err=%.3e" % err4)
results.append(("Test 4 (0 deg, 0.5)", p4))

# -- Test 5: Single point at origin ------------------------------------------
print("\nTest 5: Single point at origin")
x5 = np.array([0.0])
y5 = np.array([0.0])
err5 = roundtrip_error(x5, y5, 30.0, 0.4)
p5 = check("Roundtrip error < 1e-12", err5 < ROUNDTRIP_TOL, "max_err=%.3e" % err5)
results.append(("Test 5 (origin)", p5))

# -- Test 6: Collinear points along a line -----------------------------------
print("\nTest 6: Collinear points along y = 2x + 5")
t6 = np.linspace(0, 500, 50)
x6 = t6
y6 = 2.0 * t6 + 5.0
err6 = roundtrip_error(x6, y6, 60.0, 0.6)
p6 = check("Roundtrip error < 1e-12", err6 < ROUNDTRIP_TOL, "max_err=%.3e" % err6)
results.append(("Test 6 (collinear)", p6))

# -- Additional check A: angle=0 deg (North), ratio=0.5 ---------------------
# Azimuth 0 deg -> arithmetic 90 deg -> R rotates 90 deg CCW
# R = [[cos90, -sin90],[sin90, cos90]] = [[0, -1],[1, 0]]
# Forward: [x_c, y_c] @ R = [y_c, -x_c], then scale Y by 2
# Result: xp = y_centered, yp = -x_centered * 2
print("\nAdditional Check A: angle=0 deg (North), ratio=0.5")
print("  Azimuth 0 deg -> arithmetic 90 deg -> R rotates 90 deg CCW")
print("  Forward: xp = y_centered, yp = -x_centered * 2")
xa = np.array([100.0, 200.0, 300.0])
ya = np.array([100.0, 200.0, 300.0])
params_a = get_transform_params(xa, ya, 0.0, 0.5)
xpa, ypa = apply_transform(xa, ya, params_a)
# After centering (mean=200), x_centered = [-100, 0, 100], y_centered same
x_centered = xa - np.mean(xa)
y_centered = ya - np.mean(ya)
expected_xp = y_centered * 1.0
expected_yp = -x_centered * 2.0
err_xa = np.max(np.abs(xpa - expected_xp))
err_ya = np.max(np.abs(ypa - expected_yp))
pa1 = check("xp = y_centered (rotation maps Y->X)", err_xa < 1e-12,
            "max_err=%.3e" % err_xa)
pa2 = check("yp = -x_centered * 2 (rotation maps X->-Y, scaled by 2)", err_ya < 1e-12,
            "max_err=%.3e" % err_ya)
results.append(("Check A (az=0, rotation+scaling)", pa1 and pa2))

# -- Additional check B: angle=90 deg (East), ratio=1.0 -> identity ---------
# Azimuth 90 deg -> arithmetic 0 deg -> R = identity
# Forward: [x_c, y_c] @ I = [x_c, y_c], S=[1,1] -> no change
print("\nAdditional Check B: angle=90 deg (East), ratio=1.0 -> identity (no rotation)")
print("  Azimuth 90 deg -> arithmetic 0 deg -> R = identity")
xb = np.array([100.0, 200.0, 300.0, 400.0])
yb = np.array([50.0,  150.0, 250.0, 350.0])
params_b = get_transform_params(xb, yb, 90.0, 1.0)
xpb, ypb = apply_transform(xb, yb, params_b)
cx = np.mean(xb)
cy = np.mean(yb)
x_centered_b = xb - cx
y_centered_b = yb - cy
expected_xpb = x_centered_b
expected_ypb = y_centered_b
err_xb = np.max(np.abs(xpb - expected_xpb))
err_yb = np.max(np.abs(ypb - expected_ypb))
pb1 = check("xp = x_centered (identity, no rotation)", err_xb < 1e-12, "max_err=%.3e" % err_xb)
pb2 = check("yp = y_centered (identity, no scaling)", err_yb < 1e-12, "max_err=%.3e" % err_yb)
results.append(("Check B (az=90, identity)", pb1 and pb2))

# -- Additional check C: multiple angles, ratio=0.5 -------------------------
print("\nAdditional Check C: roundtrip for angles 0,15,30,45,60,75,90,135,180,270 deg (azimuth)")
xc = rng.uniform(0, 500, 30)
yc = rng.uniform(0, 500, 30)
all_pass_c = True
for ang in [0, 15, 30, 45, 60, 75, 90, 135, 180, 270]:
    e = roundtrip_error(xc, yc, float(ang), 0.5)
    ok = e < ROUNDTRIP_TOL
    if not ok:
        all_pass_c = False
    print("    angle=%3d deg  max_err=%.3e  %s" % (ang, e, "PASS" if ok else "FAIL"))
results.append(("Check C (multi-angle)", all_pass_c))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("%-40s %s" % ("Test", "Result"))
print("-" * 65)
all_passed = True
for label, passed in results:
    status = "PASS" if passed else "FAIL"
    print("  %-38s %s" % (label, status))
    if not passed:
        all_passed = False
print("-" * 65)
overall = "ALL PASS" if all_passed else "SOME FAILURES -- see above"
print("  Overall: %s" % overall)
print("=" * 65)

# ---------------------------------------------------------------------------
# SVG visualisation
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Task 3.2 -- Coordinate Transformation Roundtrip V&V\n"
                 "(azimuth convention: CW from North, 0°=North)",
                 fontsize=13, fontweight="bold")

    cases = [
        (x1, y1, 90.0, 1.0, "Test 1: angle=90 deg (East), ratio=1.0\n(identity)"),
        (x2, y2, 45.0, 0.5, "Test 2: angle=45 deg (NE), ratio=0.5"),
        (x3, y3, 90.0, 0.3, "Test 3: angle=90 deg (East), ratio=0.3"),
        (x4, y4, 0.0,  0.5, "Test 4: angle=0 deg (North), ratio=0.5\n(rotation + scaling)"),
        (x6, y6, 60.0, 0.6, "Test 6: collinear, angle=60 deg, ratio=0.6"),
    ]

    for idx, (xi, yi, ang, rat, title) in enumerate(cases):
        ax = axes[idx // 3][idx % 3]
        params = get_transform_params(xi, yi, ang, rat)
        xpi, ypi = apply_transform(xi, yi, params)
        xri, yri = invert_transform_coords(xpi, ypi, params)
        err = np.max(np.abs(np.column_stack((xri - xi, yri - yi))))

        ax.scatter(xi, yi, s=18, color="steelblue", alpha=0.7, label="Original")
        ax.scatter(xpi, ypi, s=18, color="darkorange", alpha=0.7, marker="^",
                   label="Transformed")
        ax.scatter(xri, yri, s=8, color="green", alpha=0.9, marker="x",
                   label="Recovered (err=%.1e)" % err)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.tick_params(labelsize=7)

    # Last panel: error bar chart across all roundtrip tests
    ax_last = axes[1][2]
    test_labels = ["T1\n90,1.0", "T2\n45,0.5", "T3\n90,0.3",
                   "T4\n0,0.5", "T5\norigin", "T6\ncollinear"]
    errors = [
        roundtrip_error(x1, y1, 90.0, 1.0),
        roundtrip_error(x2, y2, 45.0, 0.5),
        roundtrip_error(x3, y3, 90.0, 0.3),
        roundtrip_error(x4, y4, 0.0,  0.5),
        roundtrip_error(x5, y5, 30.0, 0.4),
        roundtrip_error(x6, y6, 60.0, 0.6),
    ]
    colors = ["green" if e < ROUNDTRIP_TOL else "red" for e in errors]
    ax_last.bar(test_labels, errors, color=colors, edgecolor="black", linewidth=0.5)
    ax_last.axhline(ROUNDTRIP_TOL, color="red", linestyle="--", linewidth=1,
                    label="Tolerance %.0e" % ROUNDTRIP_TOL)
    ax_last.set_yscale("log")
    ax_last.set_title("Roundtrip Max Absolute Error\n(all bars below red line = PASS)",
                      fontsize=9)
    ax_last.set_ylabel("Max |error|", fontsize=8)
    ax_last.legend(fontsize=7)
    ax_last.tick_params(labelsize=7)

    # Orientation note as figure text
    fig.text(
        0.5, 0.01,
        "Orientation (azimuth): angle_major=0 deg -> major axis along North (+Y). "
        "angle_major=90 deg -> major axis along East (+X). "
        "Internally converts to arithmetic via theta = 90° - angle.",
        ha="center", fontsize=8, style="italic", color="dimgray"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    svg_path = os.path.join(OUTPUT_DIR, "vv_transform_roundtrip.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("\nSVG saved -> %s" % svg_path)

except Exception as exc:
    print("\n[WARNING] Could not generate SVG: %s" % exc)

# ---------------------------------------------------------------------------
# Exit code
# ---------------------------------------------------------------------------
sys.exit(0 if all_passed else 1)
