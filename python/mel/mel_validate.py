# python/mel/mel_validate.py
import json
import sys
from pathlib import Path
from jsonschema import Draft202012Validator

# repo_root = .../mel
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "spec" / "mel.schema.json"

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    _SCHEMA = json.load(f)

_VALIDATOR = Draft202012Validator(_SCHEMA)

def validate_obj(obj: dict) -> bool:
    errors = sorted(_VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    return len(errors) == 0

def validate_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return validate_obj(obj)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m mel.mel_validate <json-file>")
        sys.exit(2)
    ok = validate_file(sys.argv[1])
    print("OK" if ok else "INVALID")
    sys.exit(0 if ok else 1)
