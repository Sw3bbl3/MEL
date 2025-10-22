import sys
from pathlib import Path

# Ensure the ``mel`` package (located under ./python) is importable during tests.
ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
