from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade_dit.evaluation import evaluate_npz, save_evaluation
from cascade_dit.validation import ValidationPlan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute FID and Inception Score for an ADM-compatible sample NPZ"
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _plan_hash(samples: Path, plan_path: Path | None, parser: argparse.ArgumentParser) -> str:
    manifest_path = samples.with_suffix(".json")
    manifest_hash: str | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            parser.error(f"sample manifest is incomplete: {manifest_path}")
        value = manifest.get("sampling_plan_hash", manifest.get("validation_plan_hash"))
        if not isinstance(value, str) or not value:
            parser.error(f"sample manifest has no plan hash: {manifest_path}")
        manifest_hash = value
    if plan_path is None:
        if manifest_hash is None:
            parser.error("--plan is required for an external NPZ without a sample manifest")
        return manifest_hash
    plan_hash = ValidationPlan.load(plan_path).plan_hash
    if manifest_hash is not None and manifest_hash != plan_hash:
        parser.error("sample manifest and --plan use different validation plans")
    return plan_hash


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plan_hash = _plan_hash(args.samples, args.plan, parser)
    metrics = evaluate_npz(
        args.samples,
        args.reference,
        batch_size=args.batch_size,
        device=args.device,
        log_interval=args.log_interval,
    )
    save_evaluation(args.output, metrics, plan_hash)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
