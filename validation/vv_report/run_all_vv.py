"""
Standalone V&V runner — executes all 10 validation scripts and prints a summary.

Usage:
    python docs/validation/vv_report/run_all_vv.py

This script is for local developer use. It is NOT called by Quarto or CI.
Quarto executes each script individually via its own Python cells.
"""
import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCRIPTS = [
    "vv_variogram_models.py",
    "vv_transform_roundtrip.py",
    "vv_polynomial_drift.py",
    "vv_drift_physics.py",
    "vv_aem_single_segment.py",
    "vv_aem_scaling_consistency.py",
    "vv_wrapper_no_drift.py",
    "vv_polynomial_drift_recovery.py",
    "vv_anisotropy_consistency.py",
    "vv_loocv.py",
]

results = []
for script in SCRIPTS:
    script_path = os.path.join(SCRIPT_DIR, script)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=120
    )
    elapsed = time.time() - t0
    status = "PASS" if proc.returncode == 0 else "FAIL"
    results.append((script, status, elapsed, proc.returncode))

# Print summary table
print("\n" + "=" * 70)
print(f"{'Script':<45} {'Status':<8} {'Time':>8}")
print("=" * 70)
for script, status, elapsed, rc in results:
    print(f"{script:<45} {status:<8} {elapsed:>7.1f}s")
print("=" * 70)

n_fail = sum(1 for r in results if r[1] == "FAIL")
if n_fail == 0:
    print("\nALL 10 V&V SCRIPTS PASSED")
else:
    print(f"\n*** {n_fail} SCRIPT(S) FAILED ***")
    sys.exit(1)
