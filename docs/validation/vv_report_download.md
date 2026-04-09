# V&V Report — Download

## 📥 Direct Download

**[Download V&V Report (PDF)](../assets/vv_report.pdf)**

---

## Formal Verification & Validation Report

The UK_SSPA v2 Verification and Validation Report is a comprehensive PDF document
covering all 10 validation scripts across the core modules:

- Variogram model equations
- Coordinate transform roundtrip precision
- Polynomial drift computation and recovery
- Drift physics verification
- AEM linesink potential properties
- AEM drift scaling consistency
- **Wrapper equivalence to PyKrige** (direct comparison)
- **Anisotropy pre-transform consistency** (PyKrige comparison)
- LOOCV diagnostic metrics

The report includes test case tables, PASS/FAIL results, and diagnostic figures
for each validation script.

## Alternative Download

The latest V&V Report PDF is also available from the
[GitHub Releases](https://github.com/sspa-inc/kt3d_h2o_py/releases) page.

Each tagged release automatically generates and attaches the report.

## Re-generate locally

To regenerate the report on your own machine:

```bash
# Requires: Quarto CLI, TinyTeX, Python with project dependencies
cd docs/validation/vv_report
quarto render vv_report.qmd --to pdf