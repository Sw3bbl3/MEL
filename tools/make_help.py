#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re

COLOR = "\033[36m"
RESET = "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("makefiles", nargs="+")
    args = parser.parse_args()

    targets = []
    pattern = re.compile(r"^([a-zA-Z0-9_-]+):.*?## (.*)$")

    for path_str in args.makefiles:
        path = pathlib.Path(path_str)
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                match = pattern.match(line)
                if match:
                    targets.append(match.groups())
        except OSError:
            continue

    print("Usage: make <target>\n")
    print("Targets:")
    for name, description in sorted(targets):
        print(f"  {COLOR}{name:<18}{RESET} {description}")


if __name__ == "__main__":
    main()
