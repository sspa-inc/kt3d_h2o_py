import os
import subprocess
import sys
import time
import re

SCRIPT_DIR = os.path.abspath("docs/validation")

def run_vv(script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=120
    )
    return result

def filter_output(stdout, max_width=105):
    lines = []
    for line in stdout.splitlines():
        low = line.lower()
        if '.svg' in low and any(w in low for w in ('saved', 'plot', '->', 'path')):
            continue
        if '.png' in low and any(w in low for w in ('saved', 'plot', '->', 'path')):
            continue
        line = re.sub(r'array\[np\.float64\(([^)]+)\)', r'[\1', line)
        line = re.sub(r'np\.float64\(([^)]+)\)', r'\1', line)
        if len(line) > max_width:
            rstripped = line.rstrip()
            tail = ""
            for token in ["OK PASS", "PASS", "OK FAIL", "FAIL", "SKIP"]:
                if rstripped.endswith(token):
                    tail = "  " + token
                    line = rstripped[: len(rstripped) - len(token)].rstrip()
                    break
            if tail:
                avail = max_width - len(tail) - 3
                if len(line) > avail:
                    line = line[:avail] + "..." + tail
                else:
                    line = line + tail
            else:
                line = line[:max_width - 3] + "..."
        lines.append(line)
    return "\n".join(lines)

SCRIPTS = [
    ("vv_variogram_models.py",          "Variogram Model Equations"),
    ("vv_transform_roundtrip.py",       "Coordinate Transform Roundtrip"),
    ("vv_polynomial_drift.py",          "Polynomial Drift Computation"),
    ("vv_drift_physics.py",             "Drift Physics Verification"),
    ("vv_aem_single_segment.py",        "AEM Linesink — Single Segment"),
    ("vv_aem_scaling_consistency.py",    "AEM Drift Scaling Consistency"),
    ("vv_wrapper_no_drift.py",          "Wrapper Equivalence — No Drift"),
    ("vv_polynomial_drift_recovery.py", "Polynomial Drift Recovery"),
    ("vv_anisotropy_consistency.py",    "Anisotropy Pre-Transform Consistency"),
    ("vv_loocv.py",                     "LOOCV Diagnostic Metrics"),
]

def main():
    print("Running V&V scripts to generate markdown report...")
    exec_results = []
    outputs = {}
    
    for script_file, description in SCRIPTS:
        print(f"Running {script_file}...")
        t0 = time.time()
        proc = run_vv(script_file)
        elapsed = time.time() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"
        outputs[script_file] = filter_output(proc.stdout)
        exec_results.append((script_file, description, status, elapsed))

    with open("docs/validation/vv_report/vv_report.qmd", "r", encoding="utf-8") as f:
        qmd_content = f.read()

    # Extract sections from qmd
    # We will build the markdown manually to ensure it's clean
    
    md = [
        "# Verification and Validation Report",
        "",
        "This report presents the formal verification and validation (V&V) of the **UK_SSPA v2** universal kriging tool. The tool performs spatial interpolation of water-level surfaces using kriging with specified drift, supporting four variogram models (spherical, exponential, Gaussian, linear), polynomial drift terms up to second order, and Analytic Element Method (AEM) linesink drift for incorporating river-boundary influence.",
        "",
        "Ten independent V&V scripts were executed. Each script isolates a specific computational module and verifies its output against analytical solutions, hand-calculated reference values, or the reference implementation in PyKrige. The table below summarizes the outcome of every script.",
        "",
        "| # | V&V Module | Result |",
        "|--:|------------|:------:|"
    ]
    
    for i, (sf, desc, status, elapsed) in enumerate(exec_results, 1):
        icon = "**PASS**" if status == "PASS" else "**FAIL**"
        md.append(f"| {i} | {desc} | {icon} |")
        
    n_pass = sum(1 for r in exec_results if r[2] == "PASS")
    n_fail = sum(1 for r in exec_results if r[2] == "FAIL")
    md.append("")
    if n_fail == 0:
        md.append(f"**Result: {n_pass}/{len(exec_results)} scripts passed. All tests passed.**")
    else:
        md.append(f"**Result: {n_pass} PASS, {n_fail} FAIL out of {len(exec_results)} scripts.**")
        
    md.append("")
    
    # Now we parse the QMD file to extract the text for each section
    sections = qmd_content.split("\n# ")
    
    for section in sections[1:]: # Skip the first part before the first "# "
        lines = section.split("\n")
        title = lines[0].strip()
        
        if title in ["Executive Summary", "Test Environment"]:
            continue
            
        if title == "Coverage Gap Analysis {.unnumbered}":
            title = "Coverage Gap Analysis"
        if title == "How to Re-Run {.unnumbered}":
            title = "How to Re-Run"
            
        md.append(f"## {title}")
        
        in_python_block = False
        for line in lines[1:]:
            if line.startswith("```{python}"):
                in_python_block = True
                # Check if it's a script output block
                continue
            if in_python_block and line.startswith("```"):
                in_python_block = False
                continue
                
            if in_python_block:
                if "print_filtered(_cache[" in line:
                    script_name = re.search(r'"([^"]+)"', line).group(1)
                    md.append("```text")
                    md.append(outputs[script_name])
                    md.append("```")
                continue
                
            # Fix image paths
            if line.startswith("!["):
                # Replace ../output/ with output/
                line = line.replace("../output/", "output/")
                # Remove {width=90%}
                line = re.sub(r'\{width=[^}]+\}', '', line)
                md.append(line)
            elif line.startswith("\\appendix"):
                continue
            else:
                md.append(line)
                
    with open("docs/validation/vv_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print("Generated docs/validation/vv_report.md")

if __name__ == "__main__":
    main()
