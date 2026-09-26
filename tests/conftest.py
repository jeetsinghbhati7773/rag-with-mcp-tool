from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
parent_root = PROJECT_ROOT.parent
if str(parent_root) not in sys.path:
    sys.path.insert(0, str(parent_root))
