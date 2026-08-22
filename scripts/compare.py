from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cascade_dit.compare import compare_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare matched Cascade DDT results")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison: dict[str, Any] = compare_runs(args.first, args.second)
    rendered = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
