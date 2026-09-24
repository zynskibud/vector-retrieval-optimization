"""Put the repo root on sys.path and run from it, so `indexes.python` imports and data paths resolve.

`indexes/` has no __init__.py (it is outside this package's scope), so pytest cannot find the root itself.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
