# Critical Review of [`github_plan.md`](github_plan.md)

## Scope Assessment

[`plans/github_plan.md`](plans/github_plan.md) is not actually a GitHub publication plan. It is primarily a documentation architecture and tooling plan centered on MkDocs, Jupyter notebooks, and PDF generation. If the real objective is to get the script and documentation onto GitHub, the plan is currently mis-scoped.

A GitHub publication plan should address repository readiness, version control hygiene, remote setup, documentation publishing strategy, and deployment. The current plan addresses only one slice of that problem: a possible future documentation stack.

## Major Issues

### 1. The plan is solving the wrong first problem
The title and content focus on documentation architecture rather than GitHub onboarding.

What is missing:
- repository initialization and remote setup
- `.gitignore`
- [`README.md`](README.md)
- license selection
- first commit strategy
- branch naming and push workflow
- GitHub Pages deployment approach
- CI for tests and docs

This makes the plan incomplete for the stated goal.

### 2. The plan assumes a different repository layout than the current project
The plan refers to project `kt3d_h2o_py` and proposes a [`mkdocs.yml`](mkdocs.yml) configuration with `paths: [src]`. That does not match the current repository.

Current workspace evidence:
- core modules are at the repository root, such as [`kriging.py`](kriging.py), [`variogram.py`](variogram.py), [`drift.py`](drift.py), [`data.py`](data.py), and [`main.py`](main.py)
- documentation already exists under [`docs/`](docs)
- validation scripts already exist under [`docs/validation/`](docs/validation)

Because of that mismatch, the proposed configuration would need revision before it could work.

### 3. The plan ignores the documentation that already exists
This repository is not starting from zero. It already contains substantial documentation assets, including:
- [`docs/overview.md`](docs/overview.md)
- [`docs/quickstart.md`](docs/quickstart.md)
- [`docs/configuration.md`](docs/configuration.md)
- [`docs/workflow.md`](docs/workflow.md)
- [`docs/glossary.md`](docs/glossary.md)
- theory pages under [`docs/theory/`](docs/theory)
- API pages under [`docs/api/`](docs/api)
- examples under [`docs/examples/`](docs/examples)
- validation materials under [`docs/validation/`](docs/validation)

The plan should begin with an inventory-and-reuse approach, not a greenfield architecture approach.

### 4. Notebook migration is expensive and not justified by the GitHub goal
The plan proposes converting validation scripts such as [`docs/validation/vv_variogram_models.py`](docs/validation/vv_variogram_models.py) into notebooks.

That recommendation is weak for several reasons:
- notebooks are not required to publish code or docs on GitHub
- notebooks create noisier diffs and more maintenance overhead
- notebooks introduce execution-state issues
- the repository already has Python validation scripts and generated figures in [`docs/validation/output/`](docs/validation/output)

A lower-risk approach is to keep `.py` validation scripts as the source of truth and write markdown pages that summarize the validation logic and embed the generated SVG outputs.

### 5. Dependency guidance is underspecified
The plan says to add dependencies to either [`requirements.txt`](requirements.txt) or `pyproject.toml`, but does not choose one strategy.

Problems:
- no version pinning
- no reproducibility guidance
- ambiguous plugin choice between `mkdocs-with-pdf` and `mkdocs-pdf-export-plugin`
- no distinction between required and optional documentation dependencies

For a first GitHub publication pass, PDF export should be treated as optional and deferred unless there is a confirmed need.

### 6. The proposed [`mkdocs.yml`](mkdocs.yml) contains practical errors and risky assumptions
Several parts of the sample configuration are problematic.

#### Invalid YAML value style for `repo_url`
The plan uses markdown link syntax inside YAML:
- `repo_url: [https://github.com/sspa-inc/kt3d_h2o_py](https://github.com/sspa-inc/kt3d_h2o_py)`

That is not valid for a normal YAML string value in this context. It should be a plain URL string.

#### Incorrect code path assumption
The plan uses:
- `paths: [src]`

That does not match the current root-level module layout.

#### Theme palette assumption
The plan uses:
- `primary: custom-blue`

That is not a standard Material for MkDocs palette value unless custom theme overrides are added.

#### Mermaid fence configuration may be unnecessary
The plan includes a custom fence block for Mermaid. That should only be added if Mermaid diagrams are actually needed.

#### PDF plugin in first-pass setup is risky
The `with-pdf` plugin often adds build complexity and should not be part of the first iteration unless it is already proven locally and in CI.

### 7. The navigation proposal does not match the current docs structure
The proposed nav references pages such as [`docs/index.md`](docs/index.md), but the current docs set appears to use [`docs/overview.md`](docs/overview.md) and other existing pages.

The nav should be built from the documentation that already exists, not from a hypothetical future structure.

### 8. The plan omits GitHub-specific publishing decisions
A real GitHub plan should explicitly answer:
- Will the repository be public or private?
- Will documentation be published with GitHub Pages?
- If so, will Pages build from GitHub Actions or a `gh-pages` branch?
- What branch will be the default branch?
- What content is safe to publish?

None of those decisions are addressed.

### 9. The plan does not address repository hygiene before first push
The repository contains items that should be reviewed before publication, including:
- generated outputs under [`docs/examples/output/`](docs/examples/output)
- generated outputs under [`docs/validation/output/`](docs/validation/output)
- local cache artifacts under [`.pytest_cache/`](.pytest_cache)
- test datasets and shapefiles under [`Test/Test01/`](Test/Test01)
- local log output such as [`output.log`](output.log)

A GitHub publication plan must decide what should be committed, what should be ignored, and what may be sensitive or unnecessary.

### 10. The plan omits the minimum public-facing repository files
For a GitHub repository, the following are usually more important than advanced docs plugins:
- [`README.md`](README.md)
- `LICENSE`
- optional contribution guidance
- optional issue templates

At minimum, [`README.md`](README.md) should explain:
- what the project does
- how to install or run it
- where the docs live
- what validation or tests exist

### 11. The sequencing is backwards
The current plan starts with advanced documentation tooling before establishing repository readiness.

A safer order would be:
1. audit repository contents for publishability
2. add `.gitignore`, [`README.md`](README.md), and license
3. initialize git and connect remote
4. make a clean first commit
5. decide documentation publishing strategy
6. add minimal [`mkdocs.yml`](mkdocs.yml) if needed
7. add GitHub Pages and CI only after local builds succeed

## Recommended Replacement Plan Structure

### Phase 1 — Publishability Audit
- review repository contents for secrets, proprietary data, large binaries, and generated artifacts
- decide whether [`Test/`](Test), [`output.log`](output.log), [`.pytest_cache/`](.pytest_cache), and generated SVG outputs should be versioned
- confirm whether shapefiles and sample data are safe to publish

### Phase 2 — Repository Baseline
- add `.gitignore`
- add [`README.md`](README.md)
- add `LICENSE` if appropriate
- confirm the package/install story for root-level modules such as [`kriging.py`](kriging.py), [`variogram.py`](variogram.py), [`drift.py`](drift.py), and [`main.py`](main.py)

### Phase 3 — Documentation Strategy
- reuse the existing markdown under [`docs/`](docs)
- create a minimal [`mkdocs.yml`](mkdocs.yml) that matches the current structure
- keep validation scripts in Python under [`docs/validation/`](docs/validation)
- optionally create markdown summary pages that reference figures in [`docs/validation/output/`](docs/validation/output)
- defer notebook conversion unless there is a strong presentation need

### Phase 4 — GitHub Integration
- initialize git if needed
- create the remote repository
- push the default branch
- configure GitHub Pages if documentation hosting is desired

### Phase 5 — Automation
- add CI for tests
- add CI for docs build
- optionally add release tagging/versioning

## Most Important Corrections to the Existing Plan

1. Replace the notebook-conversion requirement with an evaluation step.
   - Instead of mandating notebook migration, say that notebooks are optional presentation artifacts.

2. Replace `paths: [src]` with the actual module layout.
   - The current repository uses root-level Python modules, not a `src/` package layout.

3. Replace markdown-link YAML with valid YAML strings.
   - [`mkdocs.yml`](mkdocs.yml) should use plain URL values.

4. Remove PDF export from the first iteration.
   - Treat PDF generation as optional after the basic docs site works.

5. Add GitHub repository setup tasks.
   - Include `.gitignore`, [`README.md`](README.md), license, remote creation, first push, and Pages deployment.

6. Build navigation from the docs that already exist.
   - Use the current [`docs/`](docs) structure rather than a hypothetical one.

## Bottom Line
[`plans/github_plan.md`](plans/github_plan.md) is usable as an early brainstorming document for documentation tooling, but it is not yet a sound plan for getting this repository and its documentation onto GitHub.

It should be rewritten as a repository publication plan first, with documentation tooling treated as one later subsection rather than the main event.
