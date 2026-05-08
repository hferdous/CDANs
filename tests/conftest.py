"""Test configuration — make ``src/`` importable without installing the package.

When the package is installed via ``pip install -e .``, this file is unnecessary.
It exists so that ``pytest`` can be run directly from a fresh checkout.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
