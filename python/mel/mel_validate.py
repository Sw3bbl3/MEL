# python/mel/mel_validate.py
import json, sys
from importlib import resources
from jsonschema import Draft202012Validator

def _load_schema():
    with resources.files("mel.data").joinpath("mel.schema.json").open("r", encoding="utf-8") as f:
        return json.load(f)

_SCHEMA = _load_schema()
_VALIDATOR = Draft202012Validator(_SCHEMA)

def validate_obj(obj: dict) -> bool:
    errors = tuple(_VALIDATOR.iter_errors(obj))
    return len(errors) == 0

def validate_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        return validate_obj(json.load(f))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m mel.mel_validate <json-file>")
        sys.exit(2)
    print("OK" if validate_file(sys.argv[1]) else "INVALID")
