# GitHub Publication — Final Implementation Plan

**Project:** UK_SSPA v2 (Water Level Kriging Tool)
**Date:** 2026-04-09
**Status:** Ready for execution

---

## Context

All documentation, V&V scripts, examples, theory pages, and API references have been completed (Phases 1–7 of the documentation implementation plan). This plan covers the remaining work: preparing the repository for GitHub publication and setting up documentation hosting.

The plan is structured as atomic subtasks that can be executed independently by a simple LLM. Each subtask has a single clear deliverable, explicit inputs, and acceptance criteria.

**Standing rule:** The `Test/` directory and all its contents are **confidential** and must **never** be committed to version control. This is non-negotiable and applies to all tasks in this plan.

---

## Phase 1 — Publishability Audit

### Task 1.1 — Inventory generated artifacts

**Objective:** Identify all generated/output files that should NOT be committed to version control.

**Action:**
1. List all files under `docs/examples/output/` — these are generated SVG plots
2. List all files under `docs/validation/output/` — these are generated SVG plots
3. Check for `output.log` in the root directory
4. Check for `.pytest_cache/` directory
5. Check for `__pycache__/` directories anywhere in the tree

**Standing decisions (already resolved):**
- `Test/` and all contents — **ALWAYS IGNORE**. Confidential data. Never commit.

**Decision required from user before proceeding:**
- Should `docs/examples/output/*.svg` be committed? (They are useful for rendered docs but are regenerable.)
- Should `docs/validation/output/*.svg` be committed? (Same consideration.)

**Acceptance:** Every generated or temporary file in the repo has been categorized as "commit" or "ignore".

---

### Task 1.2 — Check for secrets and sensitive data

**Objective:** Ensure no secrets, credentials, API keys, or proprietary data paths are present in committed files.

**Action:**
1. Search all `.py` files for patterns: `password`, `secret`, `api_key`, `token`, `credential`
2. Search all `.json` files for the same patterns
3. Search all `.md` files for absolute file paths that reveal internal directory structures (e.g., `C:\Users\`, `H:\`, `/home/`)
4. Review `config.json` for any paths that reference local machine directories or the confidential `Test/` directory
5. If `config.json` references `Test/Test01/` paths, those paths must be sanitized or made generic before committing

**Deliverable:** A list of findings (or confirmation that none exist). Any issues found must be remediated before first commit.

**Acceptance:** No secrets or sensitive local paths exist in any file that will be committed.

---

## Phase 2 — Repository Baseline Files

### Task 2.1 — Create .gitignore

**Objective:** Create a comprehensive `.gitignore` for a Python scientific project.

**File:** `.gitignore`

**Content must include:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# pytest
.pytest_cache/

# OS files
Thumbs.db
.DS_Store

# Logs
output.log
*.log

# Confidential test data — NEVER COMMIT
Test/

# Generated outputs (uncomment if you decide NOT to commit SVGs)
# docs/examples/output/*.svg
# docs/validation/output/*.svg
```

**Note:** The SVG comment lines should be uncommented if the user decides not to commit generated SVGs (per Task 1.1 decision).

**Acceptance:** `.gitignore` exists in the repo root. `Test/` is unconditionally ignored. All Python cache/build artifacts are ignored.

---

### Task 2.2 — Create README.md

**Objective:** Create the primary public-facing repository file.

**File:** `README.md`

**Structure:**

```markdown
# UK_SSPA v2 — Universal Kriging for Water Level Mapping

## What It Does
[1-2 sentences: Universal Kriging with specified drift for spatial water level interpolation, supporting polynomial drift, AEM linesink drift, geometric anisotropy, and leave-one-out cross-validation.]

## Key Features
- Spherical, exponential, Gaussian, and linear variogram models
- Linear and quadratic polynomial drift terms
- Analytic Element Method (AEM) linesink drift for river influence
- Geometric anisotropy with coordinate pre-transformation
- Leave-one-out cross-validation (LOOCV)
- Grid prediction with contour export (shapefile)
- Config-driven pipeline — no code changes needed for different sites

## Quick Start
[Refer to docs/quickstart.md content — install deps, prepare config, run main.py]

## Documentation
- [Overview](docs/overview.md)
- [Quick Start](docs/quickstart.md)
- [Configuration Reference](docs/configuration.md)
- [Workflow Reference](docs/workflow.md)
- [Data Contracts](docs/data-contracts.md)
- [Glossary](docs/glossary.md)

### Theory
- [Variogram Models](docs/theory/variogram-models.md)
- [Anisotropy](docs/theory/anisotropy.md)
- [Polynomial Drift](docs/theory/polynomial-drift.md)
- [AEM Linesink Drift](docs/theory/aem-linesink.md)

### API Reference
- [data.py](docs/api/data.md)
- [variogram.py](docs/api/variogram.md)
- [transform.py](docs/api/transform.md)
- [drift.py](docs/api/drift.md)
- [AEM_drift.py](docs/api/aem_drift.md)
- [kriging.py](docs/api/kriging.md)

### Examples
- [Ordinary Kriging](docs/examples/ex_ordinary_kriging.md)
- [Linear Drift](docs/examples/ex_linear_drift.md)
- [Linesink Drift](docs/examples/ex_linesink_drift.md)
- [Anisotropy](docs/examples/ex_anisotropy.md)

### Verification & Validation
- [Tested Behaviors](docs/validation/tested-behaviors.md)
- V&V scripts in `docs/validation/`

## Dependencies
[List from imports: numpy, pandas, geopandas, shapely, scipy, matplotlib, pykrige]

## Running Tests
```
pytest
```

## License
[To be determined — see Task 2.3]
```

**Source material:** Pull the project description from `docs/overview.md`, the quick start from `docs/quickstart.md`, and the dependency list from the import statements in the codebase.

**Acceptance:** `README.md` exists, is accurate, and all internal links resolve to existing files.

---

### Task 2.3 — Add LICENSE file

**Objective:** Add a license file.

**File:** `LICENSE`

**Action:**
1. Ask the user which license to use (MIT, Apache 2.0, BSD-3-Clause, proprietary, or none)
2. If a standard open-source license: copy the full text with the correct year and copyright holder
3. If proprietary: create a simple "All Rights Reserved" notice
4. If none: skip this task

**Acceptance:** `LICENSE` file exists (or task is explicitly skipped with documented reason).

---

### Task 2.4 — Create requirements.txt

**Objective:** Pin the project's runtime dependencies for reproducibility.

**File:** `requirements.txt`

**Action:**
1. Search all `.py` files for `import` statements to identify third-party dependencies
2. Known dependencies from the codebase:
   - `numpy`
   - `pandas`
   - `geopandas`
   - `shapely`
   - `scipy`
   - `matplotlib`
   - `pykrige`
3. Run `pip freeze` (or check installed versions) to get current version numbers
4. Write `requirements.txt` with pinned versions (e.g., `numpy>=1.24,<2.0`)
5. Optionally create a `requirements-docs.txt` for documentation dependencies (deferred to Phase 3)

**Acceptance:** `requirements.txt` exists and `pip install -r requirements.txt` installs all runtime dependencies.

---

### Task 2.5 — Clean up config.json paths

**Objective:** Ensure `config.json` does not contain machine-specific absolute paths or references to confidential test data.

**File:** `config.json`

**Action:**
1. Read `config.json`
2. Check all path values (e.g., `data_sources.observation_wells.path`, `data_sources.linesink_river.path`, `output.contour_output_path`, `output.points_output_path`)
3. If any paths are absolute (e.g., `H:\...` or `C:\...`), convert them to relative paths
4. If any paths reference `Test/Test01/` (confidential data), replace them with placeholder paths and add a comment in README explaining that users must supply their own data
5. Consider whether `config.json` should be committed as-is or as a `config.example.json` template

**Acceptance:** All paths in `config.json` are relative. No references to confidential `Test/` data remain. No machine-specific paths remain.

---

## Phase 3 — Documentation Strategy (MkDocs Setup)

### Task 3.1 — Create mkdocs.yml

**Objective:** Create a minimal, working MkDocs configuration that matches the existing docs structure.

**File:** `mkdocs.yml`

**Action:**
1. Create `mkdocs.yml` in the repo root
2. Use the `material` theme
3. Set `site_name` to `UK_SSPA v2 Documentation`
4. Set `repo_url` to the actual GitHub URL (plain string, NOT markdown link syntax)
5. Configure `mkdocstrings` with `paths: [.]` (root-level modules, NOT `src/`)
6. Do NOT include `mkdocs-jupyter` (no notebooks in this project)
7. Do NOT include `mkdocs-with-pdf` (defer PDF to a later iteration)
8. Do NOT include Mermaid custom fences unless Mermaid diagrams are confirmed in the docs
9. Use a standard Material palette value (e.g., `primary: indigo`), NOT `custom-blue`
10. Build the `nav:` section from the actual existing docs structure

**Content:**
```yaml
site_name: UK_SSPA v2 Documentation
repo_url: https://github.com/OWNER/REPO_NAME
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
  palette:
    scheme: default
    primary: indigo

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [.]

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences
  - tables
  - toc:
      permalink: true

nav:
  - Home: overview.md
  - Quick Start: quickstart.md
  - Configuration: configuration.md
  - Workflow: workflow.md
  - Data Contracts: data-contracts.md
  - Glossary: glossary.md
  - Theory:
      - Variogram Models: theory/variogram-models.md
      - Anisotropy: theory/anisotropy.md
      - Polynomial Drift: theory/polynomial-drift.md
      - AEM Linesink Drift: theory/aem-linesink.md
  - API Reference:
      - data.py: api/data.md
      - variogram.py: api/variogram.md
      - transform.py: api/transform.md
      - drift.py: api/drift.md
      - AEM_drift.py: api/aem_drift.md
      - kriging.py: api/kriging.md
  - Examples:
      - Ordinary Kriging: examples/ex_ordinary_kriging.md
      - Linear Drift: examples/ex_linear_drift.md
      - Linesink Drift: examples/ex_linesink_drift.md
      - Anisotropy: examples/ex_anisotropy.md
  - Validation:
      - Tested Behaviors: validation/tested-behaviors.md
```

**Note:** Replace `OWNER/REPO_NAME` with the actual GitHub repository path once created.

**Acceptance:** Running `mkdocs build` from the repo root succeeds without errors. Running `mkdocs serve` renders the site locally.

---

### Task 3.2 — Create docs requirements file

**Objective:** Separate documentation build dependencies from runtime dependencies.

**File:** `requirements-docs.txt`

**Content:**
```
mkdocs-material>=9.0
mkdocstrings[python]>=0.24
```

**Acceptance:** `pip install -r requirements-docs.txt` installs all docs build dependencies.

---

### Task 3.3 — Create docs index page

**Objective:** MkDocs expects an `index.md` or a configured home page. The current docs use `overview.md` as the home page.

**Action:**
1. Check if `mkdocs.yml` nav correctly maps `Home: overview.md`
2. If MkDocs requires `docs/index.md`, create it as a redirect or copy of `overview.md`
3. Alternatively, just set `nav: - Home: overview.md` which MkDocs supports directly

**Acceptance:** `mkdocs build` resolves the home page without warnings.

---

### Task 3.4 — Test local MkDocs build

**Objective:** Verify the documentation site builds cleanly.

**Action:**
1. Install docs dependencies: `pip install -r requirements-docs.txt`
2. Run `mkdocs build --strict` from the repo root
3. Fix any warnings or errors:
   - Missing pages referenced in nav
   - Broken internal links
   - Missing images or SVG references
4. Run `mkdocs serve` and visually verify the site renders correctly

**Acceptance:** `mkdocs build --strict` exits with code 0 and no warnings.

---

## Phase 4 — Git Initialization and GitHub Setup

### Task 4.1 — Initialize local git repository

**Objective:** Set up version control locally.

**Action:**
1. Run `git init` in the repo root
2. Verify `.gitignore` is in place (from Task 2.1)
3. Run `git add .` to stage all non-ignored files
4. Review staged files with `git status` — verify no unwanted files are staged
5. Specifically verify that `Test/` is NOT staged
6. Specifically verify that `.pytest_cache/` is NOT staged
7. Specifically verify that `output.log` is NOT staged
8. If unwanted files appear, update `.gitignore` and re-stage

**Acceptance:** `git status` shows only intended files staged. No confidential data, generated outputs, caches, or sensitive files are staged.

---

### Task 4.2 — Make first commit

**Objective:** Create a clean initial commit.

**Action:**
1. Run `git commit -m "Initial commit: UK_SSPA v2 kriging tool with full documentation"`
2. Verify commit succeeded

**Acceptance:** `git log --oneline` shows exactly one commit.

---

### Task 4.3 — Create GitHub remote repository

**Objective:** Set up the remote repository on GitHub.

**Action (manual or via GitHub CLI):**
1. Decide: public or private repository
2. Create the repository on GitHub (via web UI or `gh repo create`)
   - Repository name: to be determined by user
   - Description: "Universal Kriging with specified drift for spatial water level mapping"
   - Do NOT initialize with README (we already have one)
   - Do NOT add .gitignore (we already have one)
   - Do NOT add license (we already have one)
3. Copy the remote URL

**Acceptance:** Empty repository exists on GitHub.

---

### Task 4.4 — Connect local repo to GitHub remote

**Objective:** Link the local repository to the GitHub remote.

**Action:**
1. Run `git remote add origin https://github.com/OWNER/REPO_NAME.git`
2. Run `git branch -M main`
3. Run `git push -u origin main`
4. Verify push succeeded by checking GitHub web UI

**Acceptance:** All files are visible on GitHub. README renders correctly on the repository landing page.

---

### Task 4.5 — Verify GitHub rendering

**Objective:** Confirm that documentation renders correctly on GitHub.

**Action:**
1. Navigate to the repository on GitHub
2. Verify `README.md` renders with correct formatting and working links
3. Click through documentation links in README to verify they resolve
4. Check that SVG images render in the docs (if committed)
5. Check that code blocks in markdown files render with syntax highlighting

**Acceptance:** All documentation is readable and navigable directly on GitHub.

---

## Phase 5 — GitHub Pages Deployment (Optional)

### Task 5.1 — Decide deployment strategy

**Objective:** Choose how to publish the MkDocs site.

**Options:**
- **Option A: GitHub Actions** — Automated build and deploy on every push to `main`
- **Option B: Manual `gh-pages` branch** — Run `mkdocs gh-deploy` locally
- **Option C: Skip** — Documentation is readable directly on GitHub without a hosted site

**Action:** Choose one option. If Option C, skip Tasks 5.2–5.4.

**Acceptance:** Decision is documented.

---

### Task 5.2 — Create GitHub Actions workflow for docs (Option A only)

**Objective:** Automate documentation deployment.

**File:** `.github/workflows/docs.yml`

**Content:**
```yaml
name: Deploy Documentation

on:
  push:
    branches:
      - main
    paths:
      - 'docs/**'
      - 'mkdocs.yml'

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-docs.txt

      - name: Build and deploy
        run: mkdocs gh-deploy --force
```

**Acceptance:** Workflow file exists and is syntactically valid YAML.

---

### Task 5.3 — Enable GitHub Pages

**Objective:** Configure GitHub Pages to serve from the `gh-pages` branch.

**Action:**
1. Go to repository Settings → Pages
2. Set Source to "Deploy from a branch"
3. Set Branch to `gh-pages` / `/ (root)`
4. Save

**Acceptance:** GitHub Pages is enabled and shows the deployment source.

---

### Task 5.4 — Verify deployed documentation site

**Objective:** Confirm the hosted site works.

**Action:**
1. Trigger a docs build (push a change to `docs/` or `mkdocs.yml`, or run `mkdocs gh-deploy` locally)
2. Wait for GitHub Actions to complete (if using Option A)
3. Navigate to `https://OWNER.github.io/REPO_NAME/`
4. Verify:
   - Home page loads
   - Navigation works
   - All pages render
   - SVG images display
   - Search works
   - LaTeX equations render (if any)

**Acceptance:** Documentation site is live and fully functional.

---

## Phase 6 — CI for Tests (Optional)

### Task 6.1 — Create GitHub Actions workflow for tests

**Objective:** Run the test suite on every push.

**File:** `.github/workflows/tests.yml`

**Content:**
```yaml
name: Run Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest

      - name: Run tests
        run: pytest --tb=short -q
```

**Note:** Tests that depend on `Test/Test01/` data (e.g., `test_real_data.py`) must gracefully skip when the data is absent. Verify this works in CI where `Test/` will not exist.

**Acceptance:** Workflow file exists and is syntactically valid YAML.

---

### Task 6.2 — Verify CI pipeline

**Objective:** Confirm tests pass in CI.

**Action:**
1. Push the workflow file to GitHub
2. Check the Actions tab for the test run
3. Verify all tests pass (63 passed, 1 skipped, 0 failed — matching local results)
4. The skip should be `test_real_data.py` which depends on confidential `Test/` data that is not in the repo

**Acceptance:** CI badge shows green. All tests pass or skip gracefully.

---

### Task 6.3 — Add CI status badge to README

**Objective:** Show test status on the repository landing page.

**File:** `README.md` (update)

**Action:** Add at the top of README.md:
```markdown
![Tests](https://github.com/OWNER/REPO_NAME/actions/workflows/tests.yml/badge.svg)
```

**Acceptance:** Badge renders on GitHub and reflects current test status.

---

## Phase 7 — Final Cleanup

### Task 7.1 — Review all markdown links

**Objective:** Ensure all internal links in documentation work on GitHub.

**Action:**
1. Search all `.md` files for relative links (pattern: `](`)
2. Verify each linked file exists at the referenced path
3. Fix any broken links
4. Pay special attention to links that use line-number anchors (e.g., `file.py:42`) — these work in some contexts but not in GitHub rendered markdown
5. Ensure no links point to files inside `Test/` (which will not be in the repo)

**Acceptance:** No broken links in any `.md` file when viewed on GitHub.

---

### Task 7.2 — Update mkdocs.yml repo_url

**Objective:** Replace placeholder URL with actual repository URL.

**File:** `mkdocs.yml`

**Action:** Replace `https://github.com/OWNER/REPO_NAME` with the actual repository URL.

**Acceptance:** `repo_url` in `mkdocs.yml` points to the correct GitHub repository.

---

### Task 7.3 — Update activeContext.md

**Objective:** Record the completion of GitHub publication.

**File:** `activeContext.md`

**Action:** Add a summary entry documenting:
- GitHub repository URL
- Whether GitHub Pages is enabled
- Whether CI is enabled
- Decisions made: Test/ excluded (confidential), license choice, SVG inclusion decision

**Acceptance:** `activeContext.md` reflects the current project state.

---

## Execution Order Summary

```
Phase 1 (Audit)
  1.1 Inventory artifacts
  1.2 Check for secrets
      ↓
Phase 2 (Baseline Files)
  2.1 .gitignore  (depends on 1.1)
  2.2 README.md
  2.3 LICENSE
  2.4 requirements.txt
  2.5 Clean config.json paths  (depends on 1.2)
      ↓
Phase 3 (MkDocs)
  3.1 mkdocs.yml
  3.2 requirements-docs.txt
  3.3 docs index page
  3.4 Test local build  (depends on 3.1, 3.2, 3.3)
      ↓
Phase 4 (Git + GitHub)
  4.1 git init  (depends on 2.1)
  4.2 First commit  (depends on 4.1)
  4.3 Create remote
  4.4 Push  (depends on 4.2, 4.3)
  4.5 Verify rendering  (depends on 4.4)
      ↓
Phase 5 (GitHub Pages — optional)
  5.1 Choose strategy
  5.2 Actions workflow  (depends on 5.1)
  5.3 Enable Pages  (depends on 5.2)
  5.4 Verify site  (depends on 5.3)
      ↓
Phase 6 (CI — optional)
  6.1 Test workflow
  6.2 Verify CI  (depends on 6.1)
  6.3 Add badge  (depends on 6.2)
      ↓
Phase 7 (Cleanup)
  7.1 Review links  (depends on 4.5)
  7.2 Update repo URL  (depends on 4.3)
  7.3 Update activeContext.md  (depends on all)
```

---

## Task Summary Table

| Task | Phase | Deliverable | Depends On | Requires User Input |
|------|-------|-------------|------------|---------------------|
| 1.1 | Audit | Artifact inventory | — | Yes (SVG decision) |
| 1.2 | Audit | Secrets check | — | No |
| 2.1 | Baseline | `.gitignore` | 1.1 | No |
| 2.2 | Baseline | `README.md` | — | No |
| 2.3 | Baseline | `LICENSE` | — | Yes (license choice) |
| 2.4 | Baseline | `requirements.txt` | — | No |
| 2.5 | Baseline | Clean `config.json` | 1.2 | No |
| 3.1 | MkDocs | `mkdocs.yml` | 2.2 | No |
| 3.2 | MkDocs | `requirements-docs.txt` | — | No |
| 3.3 | MkDocs | Index page | 3.1 | No |
| 3.4 | MkDocs | Local build test | 3.1, 3.2, 3.3 | No |
| 4.1 | Git | `git init` | 2.1 | No |
| 4.2 | Git | First commit | 4.1 | No |
| 4.3 | Git | Remote repo | — | Yes (public/private, name) |
| 4.4 | Git | Push | 4.2, 4.3 | No |
| 4.5 | Git | Verify rendering | 4.4 | No |
| 5.1 | Pages | Strategy decision | 4.4 | Yes |
| 5.2 | Pages | `.github/workflows/docs.yml` | 5.1 | No |
| 5.3 | Pages | Enable Pages | 5.2 | No |
| 5.4 | Pages | Verify site | 5.3 | No |
| 6.1 | CI | `.github/workflows/tests.yml` | 4.4 | No |
| 6.2 | CI | Verify CI | 6.1 | No |
| 6.3 | CI | Badge in README | 6.2 | No |
| 7.1 | Cleanup | Link review | 4.5 | No |
| 7.2 | Cleanup | Update repo URL | 4.3 | No |
| 7.3 | Cleanup | Update activeContext.md | All | No |

---

## Decisions Required Before Execution

Before starting, the following decisions must be made:

1. **SVG outputs:** Commit `docs/examples/output/*.svg` and `docs/validation/output/*.svg` to the repo? (Recommended: Yes — they are small and needed for rendered docs)
2. **License:** Which license? (MIT, Apache 2.0, BSD-3-Clause, proprietary, or none)
3. **Repository visibility:** Public or private?
4. **Repository name:** What should the GitHub repository be named?
5. **GitHub Pages:** Deploy documentation site, or rely on GitHub's built-in markdown rendering?
6. **CI:** Set up automated testing, or defer?

**Already decided:**
- `Test/` directory — **NEVER COMMIT** (confidential data)
- Notebook conversion — **NOT NEEDED** (validation scripts stay as `.py`)
- PDF export — **DEFERRED** (not part of first iteration)
