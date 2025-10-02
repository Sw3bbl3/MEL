from pathlib import Path
import json
from mel.mel_validate import validate_obj

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
EXAMPLES_DIR = REPO_ROOT / "spec" / "examples"

EXAMPLES = [
    EXAMPLES_DIR / "qa_request.json",
    EXAMPLES_DIR / "qa_result.json",
    EXAMPLES_DIR / "detect_request.json",
]

def test_examples_validate():
    for p in EXAMPLES:
        assert p.exists(), f"Missing example file: {p}"
        # tolerate UTF-8 BOM in example files
        obj = json.loads(p.read_text(encoding="utf-8-sig"))
        assert validate_obj(obj), f"Schema failed for {p}"
