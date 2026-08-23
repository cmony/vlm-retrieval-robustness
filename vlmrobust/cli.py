from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .analysis import analyze_output, discover_complete_models
from .config import build_condition_runs, load_config
from .runner import load_index_for_config, prepare_config, run_experiment, summarize_workload


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/a100_full.yaml", help="YAML experiment configuration")
    parser.add_argument("--models", nargs="+", help="Run only the listed configured model names")
    parser.add_argument("--output-dir", help="Override the configured output directory")
    return parser


def main_run(argv: Sequence[str] | None = None) -> int:
    parser = _common_parser("Run the COCO VLM robustness experiment")
    parser.add_argument("--limit", type=int, help="Use a deterministic subset (for example 100 or 500)")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print workload without loading models")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid existing checkpoints")
    parser.add_argument("--no-analyze", action="store_true", help="Do not generate summary CSV files after inference")
    args = parser.parse_args(argv)
    base = load_config(args.config)
    config = prepare_config(base, args.limit, args.models, args.output_dir)
    if args.dry_run:
        index = load_index_for_config(config)
        summary = summarize_workload(
            len(index.records), len(index.captions), len(build_condition_runs(config)), len(config["models"])
        )
        print(json.dumps(summary, indent=2))
        return 0
    output_dir = run_experiment(config, resume=not args.no_resume)
    if not args.no_analyze:
        analysis_config = prepare_config(base, args.limit, None, args.output_dir)
        completed_models = discover_complete_models(output_dir, analysis_config)
        paths = analyze_output(output_dir, analysis_config, completed_models)
        print("Analysis outputs:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    return 0


def main_analyze(argv: Sequence[str] | None = None) -> int:
    parser = _common_parser("Analyze completed VLM robustness rank checkpoints")
    parser.add_argument(
        "--analysis-output-dir",
        help="Write analysis CSV files outside the immutable rank-checkpoint directory",
    )
    args = parser.parse_args(argv)
    config = prepare_config(load_config(args.config), model_names=args.models, output_dir=args.output_dir)
    output_dir = Path(config["experiment"]["output_dir"])
    paths = analyze_output(output_dir, config, args.models, args.analysis_output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0
