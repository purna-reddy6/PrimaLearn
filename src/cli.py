"""
cli.py — Command-line interface for BOOTSTRAP experiments.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_info(args):
    """Print library info."""
    from src.spelke_dsl import build_spelke_library
    registry = build_spelke_library()
    print(registry.summary())
    print(f"\nCardinality by system:")
    for system, count in registry.system_cardinality().items():
        print(f"  {system}: {count}")
    print(f"\nTotal: {registry.total_cardinality()} primitives")


def cmd_compare(args):
    """Compare Spelke vs generic DSL cardinality."""
    from src.spelke_dsl import build_spelke_library
    from src.baselines.generic_dsl import build_generic_dsl

    spelke = build_spelke_library()
    generic = build_generic_dsl()

    print("Library Comparison")
    print("=" * 50)
    print(f"  Spelke DSL: {spelke.total_cardinality()} primitives")
    for system, count in spelke.system_cardinality().items():
        print(f"    {system}: {count}")
    print(f"\n  Generic DSL: {generic.total_cardinality()} primitives")
    print(f"\n  Cardinality difference: {spelke.total_cardinality() - generic.total_cardinality()}")


def cmd_run(args):
    """Run a wake-sleep experiment."""
    from src.spelke_dsl import build_spelke_library
    from src.engine.library import Library
    from src.engine.wake_sleep import WakeSleepEngine, WakeSleepConfig
    from src.arc.loader import ArcDataset

    # Load dataset
    print(f"Loading ARC dataset from {args.data_dir}...")
    dataset = ArcDataset(args.data_dir)
    tasks = dataset.load_split(args.split)
    print(f"Loaded {len(tasks)} tasks")

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
        print(f"Using first {len(tasks)} tasks")

    # Build library
    registry = build_spelke_library()
    print(f"Built Spelke library: {registry.total_cardinality()} primitives")

    # Configure engine
    config = WakeSleepConfig(
        n_iterations=args.iterations,
        search_timeout_per_task=args.timeout,
        max_search_attempts=args.max_attempts,
        checkpoint_dir=args.output_dir,
        verbose=True,
    )

    # Run
    engine = WakeSleepEngine(config)
    engine.initialize(registry)
    results = engine.run(tasks)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = [r.to_dict() for r in results]
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(engine.summary())


def cmd_analyze(args):
    """Analyze experiment results for Carey signature."""
    from src.analysis.carey_signature import analyze_carey_signature
    from src.engine.library import Library
    from src.spelke_dsl import build_spelke_library

    # Load library from checkpoint
    registry = build_spelke_library()
    library = Library(registry)

    # Load abstractions from results
    results_path = Path(args.results_dir) / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        print(f"Loaded results from {results_path}")

    report = analyze_carey_signature(library)
    print(report.summary())


def main():
    parser = argparse.ArgumentParser(
        prog="bootstrap",
        description="BOOTSTRAP: Spelke-Initialized Library Learning for ARC-AGI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # info
    sub = subparsers.add_parser("info", help="Print Spelke library info")
    sub.set_defaults(func=cmd_info)

    # compare
    sub = subparsers.add_parser("compare", help="Compare Spelke vs generic DSL")
    sub.set_defaults(func=cmd_compare)

    # run
    sub = subparsers.add_parser("run", help="Run wake-sleep experiment")
    sub.add_argument("--data-dir", default="data/arc-agi-1", help="ARC dataset directory")
    sub.add_argument("--split", default="training", help="Dataset split")
    sub.add_argument("--iterations", type=int, default=5, help="Wake-sleep iterations")
    sub.add_argument("--timeout", type=float, default=30.0, help="Search timeout per task (s)")
    sub.add_argument("--max-attempts", type=int, default=500, help="Max search attempts per task")
    sub.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    sub.add_argument("--output-dir", default="experiments/outputs/run1", help="Output directory")
    sub.set_defaults(func=cmd_run)

    # analyze
    sub = subparsers.add_parser("analyze", help="Analyze results for Carey signature")
    sub.add_argument("--results-dir", required=True, help="Results directory")
    sub.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
