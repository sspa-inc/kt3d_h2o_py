# Documentation Architecture & Implementation Plan
**Project:** `kt3d_h2o_py`
**Framework:** Material for MkDocs + Jupyter Integration

## 1. Environment & Dependency Setup
We will use an entirely free, Python-native documentation stack.

**Action Items:**
* Add the following documentation dependencies to your `requirements.txt` or `pyproject.toml` (under a `[project.optional-dependencies]` docs group):
    ```text
    mkdocs-material
    mkdocstrings[python]
    mkdocs-jupyter
    mkdocs-with-pdf  # Or mkdocs-pdf-export-plugin
    ```
* Install them locally: `pip install -e .[docs]` (if using pyproject.toml).

## 2. V&V Script Migration (The "Shift")
To make the V&V section read like a formal report, we will shift the Phase 3 validation scripts into Jupyter Notebooks.

**Action Items:**
* Create a new directory: `docs/validation/notebooks/`.
* Convert scripts like `vv_variogram_models.py` into `vv_variogram_models.ipynb`.
* **Structure each notebook as follows:**
    1.  **Markdown Cell:** Objective and analytical math/theory being tested.
    2.  **Code Cell:** Execution of the kriging/drift functions.
    3.  **Code Cell:** Assertion checks (the PASS/FAIL logic).
    4.  **Code Cell:** Matplotlib generation (the output will render directly below the cell in the final docs).

## 3. Configuration Setup (`mkdocs.yml`)
Create the `mkdocs.yml` file in the root of the `kt3d_h2o_py` repository. This acts as the central nervous system for the documentation site.

**Action Items:**
* Implement the following configuration skeleton:

```yaml
site_name: UK_SSPA v2 Documentation
repo_url: [https://github.com/sspa-inc/kt3d_h2o_py](https://github.com/sspa-inc/kt3d_h2o_py)
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
  palette:
    scheme: default
    primary: custom-blue # Adjust to company branding

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src] # Adjust if your code is in a different root dir
  - mkdocs-jupyter:
      execute: false # Set to true only if you want MkDocs to run the notebooks on build. Usually safer to run them locally/in CI and just render the outputs.
  - with-pdf:
      cover: true
      toc_title: "Table of Contents"

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true # For rendering LaTeX equations in the theory sections
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

nav:
  - Home: index.md
  - Quickstart: quickstart.md
  - Theory:
      - Variogram Models: theory/variogram-models.md
      - Anisotropy: theory/anisotropy.md
  - API Reference:
      - Data Loading: api/data.md
      - Kriging: api/kriging.md
  - Verification & Validation:
      - Variogram Validation: validation/notebooks/vv_variogram_models.ipynb
      - Polynomial Drift: validation/notebooks/vv_polynomial_drift.ipynb
