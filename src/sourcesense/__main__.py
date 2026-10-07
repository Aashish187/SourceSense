from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sourcesense",
        description="SourceSense synthetic purifier foundation.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    return parser


def main() -> int:
    parser = build_parser()
    _ = parser.parse_args()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
