import sys, json
from pathlib import Path
from .mel_validate import validate_obj

def lint_path(p: Path) -> int:
    if p.is_file() and p.suffix == ".json":
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
            ok = validate_obj(obj)
            print(f"{'OK     ' if ok else 'INVALID'} {p}")
            return 0 if ok else 1
        except Exception as e:
            print(f"ERROR  {p}  {e}")
            return 2
    if p.is_dir():
        codes = [lint_path(pp) for pp in p.rglob("*.json")]
        return 0 if all(c == 0 for c in codes) else 1
    print(f"SKIP   {p}")
    return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: mel-lint <file-or-dir> [...]")
        sys.exit(2)
    code = 0
    for arg in sys.argv[1:]:
        code |= lint_path(Path(arg))
    sys.exit(code)
