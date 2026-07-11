"""Ensures the project root is importable as ``src``/``app`` regardless of where pytest
is invoked from — same pattern used in the notebooks and app/streamlit_app.py, since this
project isn't pip-installed in editable mode.
"""

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found in any parent)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
