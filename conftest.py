"""
conftest.py — pytest configuration for CI compatibility.

The codebase uses `from v2_Code.module import ...` style imports, which require
the parent directory of this repo to be on sys.path. This conftest adds a
`v2_Code` alias pointing to the repo root so those imports resolve correctly
regardless of where the repo is checked out.
"""
import sys
import os
from pathlib import Path

# The repo root (where this conftest.py lives)
REPO_ROOT = Path(__file__).parent.resolve()

# Add repo root to sys.path so plain `import data` etc. also work
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add a v2_Code entry pointing to the repo root so that
# `from v2_Code.data import ...` resolves to `./data.py`
import importlib
import types

# Create a synthetic v2_Code package that maps to the repo root modules
if "v2_Code" not in sys.modules:
    v2_pkg = types.ModuleType("v2_Code")
    v2_pkg.__path__ = [str(REPO_ROOT)]
    v2_pkg.__package__ = "v2_Code"
    sys.modules["v2_Code"] = v2_pkg
