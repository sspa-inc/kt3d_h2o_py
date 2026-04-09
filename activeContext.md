# Active Context

## Last completed task
GitHub Pages updated with latest documentation and V&V validation script changes (committed and pushed 26 modified files including README.md, docs/index.md, mkdocs.yml, V&V scripts, and generated SVG outputs; GitHub Actions will auto-deploy to https://sspa-inc.github.io/kt3d_h2o_py/).

### Files modified:

**Core code:**
- `transform.py` — Azimuth→arithmetic conversion (`theta = radians(90 - angle_deg)`), forward uses `coords @ R`, inverse uses `coords @ R.T`
- `variogram.py` — Azimuth→arithmetic conversion in `calculate_variogram_at_vector()`
- `config.json` — `angle_major`: 10.0 → 80.0 (same physical direction, new convention)
- `kriging.py` — Azimuth→arithmetic conversion before passing to PyKrige; also fixed import fallbacks
- `anisotropy_check.py` — Azimuth→arithmetic conversion for plotting

**Tests (7 fixes for pre-existing issues + convention update):**
- `test_anisotropy_transformation.py` — Added azimuth convention test, fixed import case
- `test_drift.py` — Fixed 12 tests for current API (compute_resc formula, tuple returns, etc.)
- `test_variogram_v2_integration.py` — Fixed import path
- `test_main.py` — Fixed mock paths and anisotropy branch
- `test_real_data.py` — Added skip for missing config

**V&V scripts:**
- `docs/validation/vv_transform_roundtrip.py` — Rewrote Check A/B expected values for azimuth; 9/9 PASS
- `docs/validation/vv_anisotropy_consistency.py` — Converted all angles to azimuth, added azimuth→arithmetic for PyKrige calls; 7/7 PASS

**Example scripts:**
- `docs/examples/ex_anisotropy.py` — Changed to ANGLE_MAJOR=30° azimuth (non-degenerate), all drawing uses ANGLE_ARITH=60°

**Documentation (.md):**
- `docs/glossary.md` — Angle convention section rewritten for azimuth
- `docs/configuration.md` — angle_major description updated, example JSON updated
- `docs/workflow.md` — Convention note updated
- `docs/theory/anisotropy.md` — Comprehensive rewrite for azimuth + KT3D alignment
- `docs/examples/ex_anisotropy.md` — Updated for 30° azimuth example
- `docs/examples/ex_linesink_drift.md` — Updated angle convention references

### Test results: 63 passed, 1 skipped, 0 failed

## Documentation progress
- All tasks through 4.4 — **DONE** (updated for azimuth convention)
- Tasks 5.1–5.6 — **DONE** (API reference docs)
- Tasks 6.2–6.3 — **DONE** (`quickstart.md`, `overview.md`)
- Tasks 4.5, 6.1 — skipped (confidential data / not needed)
- Task 7.1 — **DONE** (`docs/validation/tested-behaviors.md`)
- Saved a detailed critical review of [`plans/github_plan.md`](plans/github_plan.md) to [`plans/github_plan_review.md`](plans/github_plan_review.md), identifying scope, structure, GitHub publishing, and MkDocs configuration issues.
- Created final GitHub implementation plan at [`plans/github_implementation_plan.md`](plans/github_implementation_plan.md) — 7 phases, 25 tasks covering publishability audit, repo baseline files, MkDocs setup, git/GitHub setup, optional Pages/CI, and cleanup. `Test/` directory marked as permanently confidential/ignored.
- Task 1.1 — **DONE** (Inventoried generated artifacts; user decided to ignore all SVG files in `docs/examples/output/` and `docs/validation/output/`).
- Task 1.2 — **DONE** (Checked for secrets and sensitive data. Cleaned up `config.json` to remove references to `Test/Test01/` and use generic relative paths).
- Task 2.3 — **DONE** (Added BSD-3-Clause LICENSE file).
- Task 2.4 - **DONE** (Created `requirements.txt` with pinned versions of runtime dependencies).
- Task 3.1 - **DONE** (Created `mkdocs.yml` with MkDocs configuration).
- Task 3.2 - **DONE** (Created `requirements-docs.txt` with docs build dependencies).
- Phase 4 - **DONE** (Initialized git, made first commit, created remote repo `https://github.com/sspa-inc/kt3d_h2o_py`, pushed to main).
- Phase 5 - **DONE** (GitHub Actions workflow for docs deployment created; repo made public; GitHub Pages enabled on `gh-pages` branch; `site_description` added to `mkdocs.yml` to trigger first deployment to `https://sspa-inc.github.io/kt3d_h2o_py/`).
- Phase 6 - **DONE** (Created GitHub Actions workflow for tests, added badge to README).
- Phase 7 - **DONE** (Fixed relative links in documentation, removed line numbers from markdown links for GitHub compatibility).
- Quarto V&V Report — **DONE** (PDF-only Quarto report at docs/validation/vv_report/; CI workflow generates PDF on version tags; MkDocs has download link page; no HTML rendering to avoid MkDocs clash; all subprocess calls have error handling).
