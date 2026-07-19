"""Command-line entry point for the classroom experiment repository."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 01 command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Controlled classroom generative-agents experiment "
            "(Phase 01 repository scaffold)."
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "validate-config",
        help="Report the status of the future configuration validator.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the placeholder CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        print("Configuration validation is not implemented yet (planned for Phase 03).")
        return 0

    print("Classroom generative-agents repository scaffold is ready.")
    print("Run 'python main.py --help' to view available commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
