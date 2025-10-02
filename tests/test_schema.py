import json
from pathlib import Path
from mel.mel_validate import validate_obj

EXAMPLES = [
    Path("spec/examples/qa_request.json"),
    Path("spec/examples/qa_result.json"),
    Path("spec/examples/detect_request.json"),
]

def test_examples_validate():
    for p in EXAMPLES:
        obj = json.loads(p.read_text(encoding="utf-8"))
        assert validate_obj(obj), f"Schema failed for {p}"
