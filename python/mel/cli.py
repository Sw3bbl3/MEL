"""Command line entrypoint for MEL tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from .mel_validate import validate_obj
from .router_client import send as client_send
from .router_server import main as run_server
from .runtime import RouterRuntime


def _lint_path(path: Path) -> int:
    if path.is_file() and path.suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            print(f"ERROR  {path}  {exc}")
            return 2
        ok = validate_obj(obj)
        print(f"{'OK     ' if ok else 'INVALID'} {path}")
        return 0 if ok else 1
    if path.is_dir():
        codes = [_lint_path(p) for p in path.rglob("*.json")]
        return 0 if all(code == 0 for code in codes) else 1
    print(f"SKIP   {path}")
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    code = 0
    for target in args.targets:
        code |= _lint_path(Path(target))
    return code


def cmd_send(args: argparse.Namespace) -> int:
    request, result = client_send(
        args.text,
        intent=args.intent,
        url=args.url,
        session_id=args.session,
    )
    print(json.dumps(request.to_dict(), indent=2))
    print("\n---\n")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok() else 1


def cmd_serve(args: argparse.Namespace) -> int:
    runtime: RouterRuntime | None = None
    if args.config:
        runtime = RouterRuntime.from_config_file(args.config)
    elif args.no_defaults:
        runtime = RouterRuntime(agents=())
    run_server(host=args.host, port=args.port, runtime=runtime, config=None)
    return 0


def build_parser(prog: str = "mel") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    sub = parser.add_subparsers(dest="command", required=True)

    p_lint = sub.add_parser("lint", help="Validate MEL JSON files against the schema")
    p_lint.add_argument("targets", nargs="+", help="Files or directories to lint")
    p_lint.set_defaults(func=cmd_lint)

    p_send = sub.add_parser("send", help="Send a prompt to a running MEL router")
    p_send.add_argument("text", help="Prompt text to send")
    p_send.add_argument("--intent", default="qa", help="Intent for the task")
    p_send.add_argument("--url", default="http://127.0.0.1:8089", help="Router URL")
    p_send.add_argument("--session", help="Optional session identifier for memory continuity")
    p_send.set_defaults(func=cmd_send)

    p_serve = sub.add_parser("serve", help="Run the reference MEL router")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8089)
    p_serve.add_argument("--config", help="Path to runtime configuration (json or toml)")
    p_serve.add_argument("--no-defaults", action="store_true", help="Start with no built-in agents")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    if argv is None:
        raw_args = sys.argv[1:]
        prog = Path(sys.argv[0]).name
    else:
        raw_args = list(argv)
        prog = "mel"

    parser = build_parser(prog=prog)
    if prog == "mel-lint" and (not raw_args or raw_args[0] != "lint"):
        raw_args = ["lint", *raw_args]

    args = parser.parse_args(raw_args)
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

