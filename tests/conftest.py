"""Pytest hooks: prefer this repo's `sgtlearn` package and local extension build."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# Resolve `import sgtlearn` to the working tree (not a stale site-packages copy).
sys.path.insert(0, str(_ROOT))

_BUILD = _ROOT / "cpp" / "build"
if _BUILD.is_dir():
    # Typical layout: cpp/build/ShapeGeneralizedTrees*.so, Discretizers*.so
    sys.path.insert(0, str(_BUILD))
