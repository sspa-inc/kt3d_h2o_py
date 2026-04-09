"""
Task 3.5 — V&V: AEM Linesink Potential — Single Segment Analytical Check
=========================================================================
Verifies compute_linesink_potential() against hand-calculated values for
simple geometries.

Outputs:
  docs/validation/output/vv_aem_single_segment.svg  — comparison plots

Run from the project root:
    python docs/validation/vv_aem_single_segment.py
"""

import sys
import os
import math
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from AEM_drift import compute_linesink_potential

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((label, status, detail))
    return condition


def phi_scalar(x, y, x1, y1, x2, y2, strength=1.0):
    """Scalar wrapper around compute_linesink_potential."""
    return float(compute_linesink_potential(
        np.array([x], dtype=float),
        np.array([y], dtype=float),
        x1, y1, x2, y2, strength
    )[0])


# ---------------------------------------------------------------------------
# Hand-calculated reference for a segment along X-axis: z1=(0,0), z2=(100,0)
# strength=1.0
#
# L = 100
# mid = 50 + 0j
# half_L_vec = 50 + 0j
# ZZ = (z - 50) / 50
#
# phi = (1.0 * 100 / (4*pi)) * Re[(ZZ+1)ln(ZZ+1) - (ZZ-1)ln(ZZ-1) + 2ln(50) - 2]
# ---------------------------------------------------------------------------

def hand_calc_phi(x, y, x1=0.0, y1=0.0, x2=100.0, y2=0.0, strength=1.0):
    """
    Direct Python implementation of the AEM formula for the reference segment.
    Used as independent ground-truth for comparison.
    """
    L = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    z = complex(x, y)
    z1c = complex(x1, y1)
    z2c = complex(x2, y2)
    mid = (z1c + z2c) / 2.0
    half_L_vec = (z2c - z1c) / 2.0
    ZZ = (z - mid) / half_L_vec

    small = 1e-10
    if abs(ZZ - 1.0) < small:
        ZZ += small
    if abs(ZZ + 1.0) < small:
        ZZ = -1.0 - small

    term1 = (ZZ + 1.0) * cmath_log(ZZ + 1.0)
    term2 = (ZZ - 1.0) * cmath_log(ZZ - 1.0)
    term3 = 2.0 * cmath_log(half_L_vec)
    carg = term1 - term2 + term3 - 2.0
    phi = (strength * L / (4.0 * math.pi)) * carg.real
    return phi


def cmath_log(z):
    import cmath
    return cmath.log(z)


# ===========================================================================
# TEST CASE 1 — Segment along X-axis: specific evaluation points
# ===========================================================================
print("\n=== TEST CASE 1: Segment along X-axis (z1=(0,0), z2=(100,0), strength=1.0) ===")

# Point above midpoint
x_test, y_test = 50.0, 50.0
got = phi_scalar(x_test, y_test, 0, 0, 100, 0, 1.0)
ref = hand_calc_phi(x_test, y_test)
err = abs(got - ref)
check("TC1a: phi(50, 50) matches hand-calc", err < 1e-10,
      f"got={got:.8f}, ref={ref:.8f}, |err|={err:.2e}")

# Point below midpoint (should equal above by symmetry)
x_test2, y_test2 = 50.0, -50.0
got2 = phi_scalar(x_test2, y_test2, 0, 0, 100, 0, 1.0)
ref2 = hand_calc_phi(x_test2, y_test2)
err2 = abs(got2 - ref2)
check("TC1b: phi(50, -50) matches hand-calc", err2 < 1e-10,
      f"got={got2:.8f}, ref={ref2:.8f}, |err|={err2:.2e}")

# Point far from segment
x_far, y_far = 200.0, 0.0
got_far = phi_scalar(x_far, y_far, 0, 0, 100, 0, 1.0)
ref_far = hand_calc_phi(x_far, y_far)
err_far = abs(got_far - ref_far)
check("TC1c: phi(200, 0) matches hand-calc", err_far < 1e-10,
      f"got={got_far:.8f}, ref={ref_far:.8f}, |err|={err_far:.2e}")

# Point very near segment centerline
x_near, y_near = 50.0, 0.001
got_near = phi_scalar(x_near, y_near, 0, 0, 100, 0, 1.0)
ref_near = hand_calc_phi(x_near, y_near)
err_near = abs(got_near - ref_near)
check("TC1d: phi(50, 0.001) near centerline matches hand-calc", err_near < 1e-10,
      f"got={got_near:.8f}, ref={ref_near:.8f}, |err|={err_near:.2e}")

# ===========================================================================
# TEST CASE 2 — Symmetry: phi(x, y) == phi(x, -y)
# ===========================================================================
print("\n=== TEST CASE 2: Symmetry about segment axis ===")

# Segment centered at origin along X-axis: z1=(-50,0), z2=(50,0)
test_points = [
    (0.0, 10.0), (10.0, 5.0), (-20.0, 30.0), (60.0, 15.0), (0.0, 100.0)
]
sym_ok = True
max_sym_err = 0.0
for (px, py) in test_points:
    phi_pos = phi_scalar(px,  py, -50, 0, 50, 0, 1.0)
    phi_neg = phi_scalar(px, -py, -50, 0, 50, 0, 1.0)
    sym_err = abs(phi_pos - phi_neg)
    max_sym_err = max(max_sym_err, sym_err)
    if sym_err >= 1e-10:
        sym_ok = False

check("TC2: Symmetry phi(x,y)==phi(x,-y) for all test points",
      sym_ok and max_sym_err < 1e-10,
      f"max |phi(x,y)-phi(x,-y)| = {max_sym_err:.2e}")

# ===========================================================================
# TEST CASE 3 — Superposition: two collinear segments == one full segment
# ===========================================================================
print("\n=== TEST CASE 3: Superposition of collinear segments ===")

# Evaluate at several off-axis points
eval_x = np.array([50.0, 25.0, 75.0, 0.0, 100.0, 50.0, -10.0])
eval_y = np.array([20.0, 30.0, 15.0, 40.0,  40.0, 80.0,  25.0])

phi_full = compute_linesink_potential(eval_x, eval_y, 0, 0, 100, 0, 1.0)
phi_half1 = compute_linesink_potential(eval_x, eval_y, 0, 0, 50, 0, 1.0)
phi_half2 = compute_linesink_potential(eval_x, eval_y, 50, 0, 100, 0, 1.0)
phi_sum = phi_half1 + phi_half2

superpos_errs = np.abs(phi_full - phi_sum)
max_superpos_err = np.max(superpos_errs)
check("TC3: Superposition [0,50]+[50,100] == [0,100]",
      max_superpos_err < 1e-6,
      f"max |phi_full - phi_sum| = {max_superpos_err:.2e} (tol=1e-6, endpoint singularity)")

# ===========================================================================
# TEST CASE 4 — Zero-length segment returns zeros
# ===========================================================================
print("\n=== TEST CASE 4: Zero-length segment returns zeros ===")

eval_x4 = np.array([10.0, 20.0, 30.0])
eval_y4 = np.array([10.0, 20.0, 30.0])
phi_zero = compute_linesink_potential(eval_x4, eval_y4, 50.0, 50.0, 50.0 + 1e-8, 50.0, 1.0)
check("TC4: L < 1e-6 returns zeros",
      np.all(phi_zero == 0.0),
      f"phi_zero = {phi_zero}")

# ===========================================================================
# TEST CASE 5 — Strength linearity: phi(strength=2) == 2 * phi(strength=1)
# ===========================================================================
print("\n=== TEST CASE 5: Strength linearity ===")

eval_x5 = np.array([50.0, 10.0, 90.0, 50.0, 150.0])
eval_y5 = np.array([25.0, 40.0, 15.0, 80.0,  30.0])

phi_s1 = compute_linesink_potential(eval_x5, eval_y5, 0, 0, 100, 0, strength=1.0)
phi_s2 = compute_linesink_potential(eval_x5, eval_y5, 0, 0, 100, 0, strength=2.0)
lin_errs = np.abs(phi_s2 - 2.0 * phi_s1)
max_lin_err = np.max(lin_errs)
check("TC5: phi(strength=2) == 2*phi(strength=1)",
      max_lin_err < 1e-14,
      f"max |phi_s2 - 2*phi_s1| = {max_lin_err:.2e}")

# ===========================================================================
# RESULTS TABLE
# ===========================================================================
print("\n" + "=" * 72)
print(f"{'Test':<52} {'Status':<6} {'Detail'}")
print("=" * 72)
for label, status, detail in results:
    print(f"{label:<52} {status:<6} {detail}")
print("=" * 72)

n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print(f"\nSummary: {n_pass} PASS, {n_fail} FAIL out of {len(results)} checks")

# ===========================================================================
# SVG OUTPUT — Visualisation of test cases
# ===========================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("V&V 3.5 — AEM Linesink Potential: Single Segment Checks", fontsize=13, fontweight='bold')

# ---- Panel 1: Potential field around the reference segment ----
ax1 = axes[0, 0]
gx = np.linspace(-20, 220, 300)
gy = np.linspace(-80, 80, 200)
GX, GY = np.meshgrid(gx, gy)
gx_flat = GX.ravel()
gy_flat = GY.ravel()
phi_grid = compute_linesink_potential(gx_flat, gy_flat, 0, 0, 100, 0, 1.0).reshape(GX.shape)

cf = ax1.contourf(GX, GY, phi_grid, levels=20, cmap='RdBu_r')
fig.colorbar(cf, ax=ax1, label='phi')
ax1.plot([0, 100], [0, 0], 'k-', lw=3, label='Segment')
ax1.plot([50], [50], 'g^', ms=8, label='(50,50)')
ax1.plot([50], [-50], 'gv', ms=8, label='(50,-50)')
ax1.plot([200], [0], 'rs', ms=8, label='(200,0)')
ax1.set_title("TC1 & TC2: Potential field\n(segment z1=(0,0) to z2=(100,0))")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.legend(fontsize=7, loc='upper right')
ax1.set_xlim(-20, 220); ax1.set_ylim(-80, 80)

# ---- Panel 2: Symmetry check ----
ax2 = axes[0, 1]
y_vals = np.linspace(1, 80, 200)
phi_pos_arr = compute_linesink_potential(
    np.full_like(y_vals, 0.0), y_vals, -50, 0, 50, 0, 1.0)
phi_neg_arr = compute_linesink_potential(
    np.full_like(y_vals, 0.0), -y_vals, -50, 0, 50, 0, 1.0)
ax2.plot(y_vals, phi_pos_arr, 'b-', label='φ(0, +y)')
ax2.plot(y_vals, phi_neg_arr, 'r--', lw=2, label='φ(0, -y)')
ax2.set_title("TC2: Symmetry φ(x,y) = φ(x,−y)\n(segment centered at origin)")
ax2.set_xlabel("|y|"); ax2.set_ylabel("φ")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ---- Panel 3: Superposition ----
ax3 = axes[1, 0]
x_line = np.linspace(-20, 120, 300)
y_fixed = 20.0
phi_full_line = compute_linesink_potential(x_line, np.full_like(x_line, y_fixed), 0, 0, 100, 0, 1.0)
phi_h1_line = compute_linesink_potential(x_line, np.full_like(x_line, y_fixed), 0, 0, 50, 0, 1.0)
phi_h2_line = compute_linesink_potential(x_line, np.full_like(x_line, y_fixed), 50, 0, 100, 0, 1.0)
phi_sum_line = phi_h1_line + phi_h2_line
ax3.plot(x_line, phi_full_line, 'b-', lw=2, label='Full [0,100]')
ax3.plot(x_line, phi_sum_line, 'r--', lw=2, label='[0,50]+[50,100]')
ax3.set_title(f"TC3: Superposition at y={y_fixed}\n(max err={max_superpos_err:.2e})")
ax3.set_xlabel("x"); ax3.set_ylabel("φ")
ax3.legend(); ax3.grid(True, alpha=0.3)

# ---- Panel 4: Strength linearity ----
ax4 = axes[1, 1]
x_line4 = np.linspace(-20, 120, 300)
phi_s1_line = compute_linesink_potential(x_line4, np.full_like(x_line4, 20.0), 0, 0, 100, 0, 1.0)
phi_s2_line = compute_linesink_potential(x_line4, np.full_like(x_line4, 20.0), 0, 0, 100, 0, 2.0)
ax4.plot(x_line4, phi_s1_line, 'b-', lw=2, label='strength=1')
ax4.plot(x_line4, phi_s2_line, 'r--', lw=2, label='strength=2')
ax4.plot(x_line4, 2.0 * phi_s1_line, 'g:', lw=2, label='2 × (strength=1)')
ax4.set_title(f"TC5: Strength linearity at y=20\n(max err={max_lin_err:.2e})")
ax4.set_xlabel("x"); ax4.set_ylabel("φ")
ax4.legend(); ax4.grid(True, alpha=0.3)

# Pass/Fail annotation
status_color = 'green' if n_fail == 0 else 'red'
status_text = f"{'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'}  ({n_pass}/{len(results)})"
fig.text(0.5, 0.01, status_text, ha='center', fontsize=12,
         color=status_color, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 1])
svg_path = os.path.join(OUTPUT_DIR, 'vv_aem_single_segment.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.close()
print(f"\nSVG saved -> {svg_path}")

# Exit with non-zero code if any test failed
if n_fail > 0:
    sys.exit(1)
