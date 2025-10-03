# python/examples/chat_cli.py
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Make sure we can import from this repo's "python" folder reliably
REPO_ROOT = Path(__file__).resolve().parents[2]
PY_PATH = REPO_ROOT / "python"
if str(PY_PATH) not in sys.path:
    sys.path.insert(0, str(PY_PATH))

# Reuse your generator that emits MEL + calls the local router
from training.infer_compile import generate, validate_obj  # generate returns (req,res,full,fell_back)

BANNER = (
    "MEL Chat — local, router-backed.\n"
    "Type your question and press Enter.\n"
    "Commands: /quit  (exit),  /mel  (toggle MEL debug).\n"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-mel", action="store_true",
                    help="Print the model's generated MEL request each turn.")
    args = ap.parse_args()

    print(BANNER)
    show_mel = args.show_mel

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("/q", "/quit", "exit"):
            break
        if user.lower() == "/mel":
            show_mel = not show_mel
            print(f"(show-mel = {show_mel})")
            continue

        # For now we route everything as a QA task; you can branch on prefixes later.
        req, res, gen_text, fell_back = generate(user, intent="qa")

        if show_mel:
            # Print just the parsed MEL request object
            print("\n[MEL REQUEST]")
            print(json.dumps(req, indent=2))
            print()

        if res is None:
            print("Router unavailable or returned no JSON.")
            continue

        # Pretty print the answer (look for the 'answer' output)
        outs = {o.get("name"): o.get("value") for o in res.get("outputs", []) if isinstance(o, dict)}
        answer = outs.get("answer")
        if answer:
            print(answer)
        else:
            # fallback view – show full TASK_RESULT so you can debug
            print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
