"""
run_experiment.py — Full Spelke vs Generic DreamCoder comparison experiment.

Runs the real wake-sleep loop with:
  - Spelke-initialized DSL (131 primitives across 6 sub-libraries)
  - Generic DSL baseline (matched cardinality)
  - Type-directed enumeration
  - Stitch anti-unification compression
  - Neural recognition network (numpy MLP)

Produces: solve-rate curves, library growth, Carey signature analysis,
          sample-efficiency comparison tables.

Usage:
    cd bootstrap-substrate
    source .venv/bin/activate
    python experiments/scripts/run_experiment.py [--quick] [--cycles N] [--tasks N]
"""

from __future__ import annotations
import argparse
import json
import os
import time
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(description="Run Spelke vs Generic DreamCoder experiment")
    p.add_argument("--data-dir", default="data/arc-agi-1/data",
                   help="ARC dataset directory")
    p.add_argument("--split", default="training", choices=["training", "evaluation"],
                   help="Which ARC split to use")
    p.add_argument("--cycles", type=int, default=5,
                   help="Number of wake-sleep cycles")
    p.add_argument("--tasks", type=int, default=None,
                   help="Max tasks to use (None = all)")
    p.add_argument("--enum-cost", type=int, default=5,
                   help="Max enumeration cost (5=deep, enables cross-system Carey signal)")
    p.add_argument("--enum-budget", type=float, default=20.0,
                   help="Seconds per task for enumerator")
    p.add_argument("--no-enumerator", action="store_true",
                   help="Disable enumerator (heuristic+AST only)")
    p.add_argument("--no-neural", action="store_true",
                   help="Disable neural recognition")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 20 tasks, 2 cycles, cost=3")
    p.add_argument("--sample-efficiency", action="store_true",
                   help="Run sample-efficiency curve (n=10,25,50,100,200,400) after main experiment")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (auto-generated if not set)")
    p.add_argument("--run-generic", action="store_true",
                   help="Also run generic DSL baseline for comparison")
    p.add_argument("--run-vimrl", action="store_true",
                   help="Also run VIMRL objectness-only DSL baseline (Ainooson 2023)")
    p.add_argument("--full-spelke", action="store_true",
                   help="Use full Spelke DSL with AGENTS+PLACES (Phase 2)")
    p.add_argument("--include-persons", action="store_true",
                   help="Include PERSONS module in Spelke library (Phase 2, 6th system)")
    p.add_argument("--resume-from", default=None,
                   help="Resume from a checkpoint directory (e.g. experiments/outputs/run_X/spelke). "
                        "Loads library + solved tasks from the latest cycle and starts from the next one.")
    p.add_argument("--resume-from-cycle", type=int, default=None,
                   help="Specific cycle number to resume from (used with --resume-from). "
                        "If not set, resumes from the latest available checkpoint.")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose output")
    p.add_argument("--rust", action="store_true",
                   help="Use Rust enumerator (spelke-enumerator binary) instead of Python TypeDirectedEnumerator")
    p.add_argument("--task-dir", default=None,
                   help="Load tasks from a directory of JSON files (overrides --data-dir/--split)")
    p.add_argument("--eval-only", action="store_true",
                   help="Evaluation-only mode: load checkpoint library, run AST solver on task-dir, no wake-sleep")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint directory to load library from (used with --eval-only)")
    p.add_argument("--curriculum-pretrain", action="store_true",
                   help="Run 1 cycle on synthetic curriculum tasks first to grow the library "
                        "with cross-system abstractions, then run ARC. Loads tasks from "
                        "data/synthetic_{cross_system,number_objects,forms_objects_places,"
                        "number_objects_replicate,compounding}/")
    return p.parse_args()


def load_tasks_from_dir(task_dir: str, max_tasks: int = None):
    """Load ARC-format tasks from a directory of JSON files."""
    import numpy as np
    import glob as _glob
    from src.arc.grid import ArcTask, TaskExample, Grid
    task_files = sorted(_glob.glob(os.path.join(task_dir, "*.json")))
    if max_tasks is not None:
        task_files = task_files[:max_tasks]
    tasks = []
    errors = 0
    for tf in task_files:
        try:
            with open(tf) as f:
                data = json.load(f)
            task_id = os.path.splitext(os.path.basename(tf))[0]
            train_examples = [
                TaskExample(Grid(np.array(ex["input"])), Grid(np.array(ex["output"])))
                for ex in data["train"]
            ]
            test_examples = [
                TaskExample(Grid(np.array(ex["input"])), Grid(np.array(ex["output"])))
                for ex in data.get("test", data["train"][:1])
            ]
            tasks.append(ArcTask(task_id=task_id, train=train_examples, test=test_examples))
        except Exception as e:
            errors += 1
    if errors > 0:
        print(f"  Warning: {errors} tasks failed to load")
    return tasks


def load_tasks(data_dir: str, split: str, max_tasks: int = None):
    from src.arc.loader import ArcDataset
    dataset = ArcDataset(data_dir)
    tasks = dataset.load_split(split)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def run_spelke_experiment(tasks, config, output_dir: Path, resume_from: str = None,
                          resume_from_cycle: int = None) -> dict:
    """Run the Spelke-initialized DreamCoder loop."""
    from src.engine.wake_sleep import WakeSleepEngine
    from src.spelke_dsl import build_spelke_library
    import glob

    print(f"\n{'='*70}")
    print(f"  SPELKE-INITIALIZED DREAMCODER")
    print(f"  Tasks: {len(tasks)} | Cycles: {config.n_iterations}")
    print(f"  Enumeration: cost={config.enumeration_max_cost}, "
          f"budget={config.enumeration_budget}s, enabled={config.use_enumerator}")
    print(f"  Neural recognition: {config.use_neural_recognition}")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
        if resume_from_cycle is not None:
            print(f"  Resume from cycle: {resume_from_cycle}")
    print(f"{'='*70}")

    spelke_dir = output_dir / "spelke"
    spelke_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir = str(spelke_dir)

    engine = WakeSleepEngine(config)
    from src.spelke_dsl import build_spelke_library, build_full_spelke_library
    if hasattr(config, '_full_spelke') and config._full_spelke:
        include_persons = getattr(config, '_include_persons', False)
        registry = build_spelke_library(include_agents=True, include_places=True,
                                        include_persons=include_persons)
        tag = "AGENTS+PLACES+PERSONS" if include_persons else "AGENTS+PLACES"
        print(f"  Using FULL Spelke DSL ({tag} enabled)")
    elif hasattr(config, '_include_persons') and config._include_persons:
        registry = build_spelke_library(include_persons=True)
        print(f"  Using Spelke DSL with PERSONS enabled")
    elif hasattr(config, '_include_places') and config._include_places:
        registry = build_spelke_library(include_places=True)
        print(f"  Using Spelke DSL with PLACES enabled")
    else:
        registry = build_spelke_library()
    engine.initialize(registry)

    # Resume from checkpoint if requested
    if resume_from:
        ckpt_path = Path(resume_from)
        if resume_from_cycle is not None:
            # Use the exact cycle specified
            cycle_num = resume_from_cycle
            lib_file = ckpt_path / f"library_cycle_{cycle_num}.json"
            if not lib_file.exists():
                print(f"  WARNING: Checkpoint for cycle {cycle_num} not found at {lib_file}, starting fresh")
            else:
                print(f"  Loading checkpoint: cycle {cycle_num} from {lib_file}")
                engine.load_checkpoint(str(ckpt_path), cycle_num)
                config.n_iterations = max(1, config.n_iterations - (cycle_num + 1))
        else:
            # Find the latest cycle checkpoint in that directory
            lib_files = sorted(ckpt_path.glob("library_cycle_*.json"))
            if not lib_files:
                print(f"  WARNING: No checkpoint files found in {resume_from}, starting fresh")
            else:
                latest = lib_files[-1]
                cycle_num = int(latest.stem.split("_")[-1])
                print(f"  Loading checkpoint: cycle {cycle_num} from {latest}")
                engine.load_checkpoint(str(ckpt_path), cycle_num)
                config.n_iterations = max(1, config.n_iterations - (cycle_num + 1))

    start = time.time()
    results = engine.run(tasks)
    total_time = time.time() - start

    print(f"\n{engine.summary()}")

    # Save full results
    experiment_data = {
        "system": "spelke",
        "n_tasks": len(tasks),
        "n_cycles": config.n_iterations,
        "total_time_seconds": total_time,
        "final_solve_rate": results[-1].solve_rate if results else 0,
        "final_library_size": results[-1].total_library_size if results else 0,
        "final_n_solved": results[-1].n_solved if results else 0,
        "cycles": [r.to_dict() for r in results],
        "library_growth": [r.total_library_size for r in results],
        "solve_rates": [r.solve_rate for r in results],
        "cross_system_counts": [r.cross_system_abstractions for r in results],
        "abstractions_by_cycle": [len(r.new_abstractions) for r in results],
        "by_method": {
            "heuristic": results[-1].n_heuristic if results else 0,
            "ast": results[-1].n_ast if results else 0,
            "enumerated": results[-1].n_enumerated if results else 0,
        },
    }

    with open(spelke_dir / "experiment_results.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

    # Carey signature analysis
    carey_report = _carey_analysis(engine.library)
    with open(spelke_dir / "carey_analysis.json", "w") as f:
        json.dump(carey_report, f, indent=2)

    print(f"\n  Carey Signature Analysis:")
    print(f"    Cross-system abstractions: {carey_report['n_cross_system']}")
    for entry in carey_report["cross_system"][:5]:
        print(f"    {entry['name']}: bridges {entry['systems']}")

    return experiment_data


def run_generic_experiment(tasks, config, output_dir: Path) -> dict:
    """Run the generic DSL baseline for comparison."""
    from src.engine.wake_sleep import WakeSleepEngine, WakeSleepConfig
    from src.baselines.generic_dsl import build_generic_dsl

    print(f"\n{'='*70}")
    print(f"  GENERIC DSL BASELINE")
    print(f"  Tasks: {len(tasks)} | Cycles: {config.n_iterations}")
    print(f"{'='*70}")

    generic_dir = output_dir / "generic"
    generic_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir = str(generic_dir)

    engine = WakeSleepEngine(config)
    registry = build_generic_dsl()
    engine.initialize(registry)

    start = time.time()
    results = engine.run(tasks)
    total_time = time.time() - start

    print(f"\n{engine.summary()}")

    experiment_data = {
        "system": "generic",
        "n_tasks": len(tasks),
        "n_cycles": config.n_iterations,
        "total_time_seconds": total_time,
        "final_solve_rate": results[-1].solve_rate if results else 0,
        "final_library_size": results[-1].total_library_size if results else 0,
        "final_n_solved": results[-1].n_solved if results else 0,
        "cycles": [r.to_dict() for r in results],
        "library_growth": [r.total_library_size for r in results],
        "solve_rates": [r.solve_rate for r in results],
        "cross_system_counts": [r.cross_system_abstractions for r in results],
        "abstractions_by_cycle": [len(r.new_abstractions) for r in results],
        "by_method": {
            "heuristic": results[-1].n_heuristic if results else 0,
            "ast": results[-1].n_ast if results else 0,
            "enumerated": results[-1].n_enumerated if results else 0,
        },
    }

    with open(generic_dir / "experiment_results.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

    return experiment_data


def run_vimrl_experiment(tasks, config, output_dir: Path) -> dict:
    """Run the VIMRL objectness-only DSL baseline (Ainooson 2023)."""
    from src.engine.wake_sleep import WakeSleepEngine, WakeSleepConfig
    from src.baselines.vimrl_dsl import build_vimrl_dsl

    print(f"\n{'='*70}")
    print(f"  VIMRL OBJECTNESS-ONLY DSL BASELINE (Ainooson 2023)")
    print(f"  Tasks: {len(tasks)} | Cycles: {config.n_iterations}")
    print(f"  (Objects only — no geometry, no number, no analogy)")
    print(f"{'='*70}")

    vimrl_dir = output_dir / "vimrl"
    vimrl_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir = str(vimrl_dir)

    engine = WakeSleepEngine(config)
    registry = build_vimrl_dsl()
    engine.initialize(registry)

    start = time.time()
    results = engine.run(tasks)
    total_time = time.time() - start

    print(f"\n{engine.summary()}")

    experiment_data = {
        "system": "vimrl",
        "n_tasks": len(tasks),
        "n_cycles": config.n_iterations,
        "total_time_seconds": total_time,
        "final_solve_rate": results[-1].solve_rate if results else 0,
        "final_library_size": results[-1].total_library_size if results else 0,
        "final_n_solved": results[-1].n_solved if results else 0,
        "cycles": [r.to_dict() for r in results],
        "library_growth": [r.total_library_size for r in results],
        "solve_rates": [r.solve_rate for r in results],
        "cross_system_counts": [r.cross_system_abstractions for r in results],
        "abstractions_by_cycle": [len(r.new_abstractions) for r in results],
        "by_method": {
            "heuristic": results[-1].n_heuristic if results else 0,
            "ast": results[-1].n_ast if results else 0,
            "enumerated": results[-1].n_enumerated if results else 0,
        },
    }

    with open(vimrl_dir / "experiment_results.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

    return experiment_data


def _carey_analysis(library) -> dict:
    """Analyze the library for Carey bootstrapping signatures."""
    all_abs = library.abstractions
    cross = library.cross_system_abstractions()

    report = {
        "total_abstractions": len(all_abs),
        "n_cross_system": len(cross),
        "cross_system": [],
        "by_cycle": {},
        "most_reused": [],
        "system_pair_counts": {},
    }

    for a in cross:
        report["cross_system"].append({
            "name": a.name,
            "systems": sorted(a.systems_composed),
            "reuse_count": a.reuse_count,
            "invention_cycle": a.invention_cycle,
            "mdl_savings": a.mdl_savings,
        })

    by_cycle = library.abstractions_by_cycle()
    for cycle, abs_list in by_cycle.items():
        report["by_cycle"][str(cycle)] = {
            "count": len(abs_list),
            "cross_system": sum(1 for a in abs_list if a.is_cross_system),
            "names": [a.name for a in abs_list],
        }

    for a in library.most_reused(10):
        report["most_reused"].append({
            "name": a.name,
            "reuse_count": a.reuse_count,
            "systems": sorted(a.systems_composed),
            "mdl_savings": a.mdl_savings,
        })

    # Count system pairs that co-appear in abstractions
    from collections import Counter
    pair_counts = Counter()
    for a in all_abs:
        systems = sorted(a.systems_composed)
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                pair_counts[f"{systems[i]}+{systems[j]}"] += 1
    report["system_pair_counts"] = dict(pair_counts.most_common())

    return report


def print_comparison(spelke_data: dict, generic_data: dict = None) -> None:
    """Print a comparison table between Spelke and Generic."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT RESULTS")
    print(f"{'='*70}")

    def fmt_rate(r):
        return f"{r:.1%}"

    cycles_s = spelke_data["cycles"]
    print(f"\n  Spelke-Initialized DreamCoder:")
    print(f"  {'Cycle':<8} {'Solved':<10} {'Rate':<8} {'Library':<10} "
          f"{'New Abs':<10} {'Cross-Sys':<10} {'H/AST/Enum'}")
    print(f"  {'-'*75}")
    for r in cycles_s:
        print(f"  {r['cycle']:<8} {r['n_solved']:<10} {fmt_rate(r['solve_rate']):<8} "
              f"{r['total_library_size']:<10} {len(r['new_abstractions']):<10} "
              f"{r['cross_system_abstractions']:<10} "
              f"{r['n_heuristic']}/{r['n_ast']}/{r['n_enumerated']}")

    if generic_data:
        cycles_g = generic_data["cycles"]
        print(f"\n  Generic DSL Baseline:")
        print(f"  {'Cycle':<8} {'Solved':<10} {'Rate':<8} {'Library':<10} {'New Abs'}")
        print(f"  {'-'*50}")
        for r in cycles_g:
            print(f"  {r['cycle']:<8} {r['n_solved']:<10} {fmt_rate(r['solve_rate']):<8} "
                  f"{r['total_library_size']:<10} {len(r['new_abstractions'])}")

        # Delta
        s_final = spelke_data["final_solve_rate"]
        g_final = generic_data["final_solve_rate"]
        delta = s_final - g_final
        print(f"\n  Spelke vs Generic: {fmt_rate(s_final)} vs {fmt_rate(g_final)} "
              f"(Δ = {delta:+.1%})")
        if delta > 0:
            print(f"  ✓ Spelke initialization provides {delta:.1%} improvement")
        else:
            print(f"  ✗ No advantage (more cycles/tasks needed to see effect)")

    print(f"\n  Final: {spelke_data['final_n_solved']}/{spelke_data['n_tasks']} solved "
          f"({fmt_rate(spelke_data['final_solve_rate'])})")
    print(f"  Library growth: {spelke_data['cycles'][0]['total_library_size'] if cycles_s else 131}"
          f" → {spelke_data['final_library_size']}")
    print(f"  Total time: {spelke_data['total_time_seconds']:.1f}s")


def save_summary(output_dir: Path, spelke_data: dict, generic_data: dict = None,
                 args=None) -> None:
    """Save a human-readable summary markdown."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Experiment Results — {ts}",
        "",
        "## Configuration",
        f"- Tasks: {spelke_data['n_tasks']}",
        f"- Cycles: {spelke_data['n_cycles']}",
        "",
        "## Spelke-Initialized DreamCoder",
        "",
        f"| Cycle | Solved | Rate | Library | New Abs | Cross-Sys |",
        f"|-------|--------|------|---------|---------|-----------|",
    ]
    for r in spelke_data["cycles"]:
        lines.append(
            f"| {r['cycle']} | {r['n_solved']}/{r['n_tasks']} | "
            f"{r['solve_rate']:.1%} | {r['total_library_size']} | "
            f"{len(r['new_abstractions'])} | {r['cross_system_abstractions']} |"
        )

    if generic_data:
        lines += [
            "",
            "## Generic DSL Baseline",
            "",
            f"| Cycle | Solved | Rate | Library |",
            f"|-------|--------|------|---------|",
        ]
        for r in generic_data["cycles"]:
            lines.append(
                f"| {r['cycle']} | {r['n_solved']}/{r['n_tasks']} | "
                f"{r['solve_rate']:.1%} | {r['total_library_size']} |"
            )

        s_final = spelke_data["final_solve_rate"]
        g_final = generic_data["final_solve_rate"]
        lines += [
            "",
            f"## Comparison",
            f"- Spelke: **{s_final:.1%}**",
            f"- Generic: **{g_final:.1%}**",
            f"- Delta: **{s_final - g_final:+.1%}**",
        ]

    lines += [
        "",
        "## Solve Method Breakdown",
        f"- Heuristic: {spelke_data['by_method']['heuristic']}",
        f"- AST solver: {spelke_data['by_method']['ast']}",
        f"- Enumerator: {spelke_data['by_method']['enumerated']}",
    ]

    with open(output_dir / "RESULTS.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Results saved to: {output_dir}/")
    print(f"  Summary: {output_dir}/RESULTS.md")


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.tasks = args.tasks or 20
        args.cycles = args.cycles if args.cycles != 5 else 2
        args.enum_cost = args.enum_cost if args.enum_cost != 5 else 3
        args.enum_budget = args.enum_budget if args.enum_budget != 20.0 else 2.0
        print("  [Quick mode: 20 tasks, 2 cycles, cost=3]")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quick" if args.quick else "full"
        output_dir = Path("experiments/outputs") / f"run_{mode}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks from custom dir or ARC dataset
    if getattr(args, 'task_dir', None):
        print(f"\nLoading tasks from {args.task_dir}...")
        tasks = load_tasks_from_dir(args.task_dir, args.tasks)
        print(f"Loaded {len(tasks)} tasks from {args.task_dir}")
    else:
        print(f"\nLoading ARC tasks from {args.data_dir}/{args.split}...")
        try:
            tasks = load_tasks(args.data_dir, args.split, args.tasks)
        except Exception as e:
            print(f"Error loading tasks: {e}")
            print("Make sure the ARC dataset is cloned:")
            print("  git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/arc-agi-1")
            sys.exit(1)

    print(f"Loaded {len(tasks)} tasks")

    # Build config
    from src.engine.wake_sleep import WakeSleepConfig
    config = WakeSleepConfig(
        n_iterations=args.cycles,
        search_timeout_per_task=3.0,
        max_search_attempts=500,
        enumeration_max_cost=args.enum_cost,
        enumeration_budget=args.enum_budget,
        use_enumerator=not args.no_enumerator,
        use_rust_enumerator=getattr(args, 'rust', False),
        use_neural_recognition=not args.no_neural,
        dream_samples_per_cycle=10,
        max_abstractions_per_cycle=15,
        verbose=args.verbose,
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "n_tasks": len(tasks),
            "n_cycles": args.cycles,
            "enum_cost": args.enum_cost,
            "enum_budget": args.enum_budget,
            "use_enumerator": not args.no_enumerator,
            "use_rust_enumerator": getattr(args, 'rust', False),
            "use_neural": not args.no_neural,
            "quick": args.quick,
        }, f, indent=2)

    # ── Eval-only mode: load checkpoint library, run AST solver, no wake-sleep ──
    if getattr(args, 'eval_only', False):
        print(f"\n{'='*70}")
        print(f"  EVAL-ONLY MODE")
        print(f"  Tasks: {len(tasks)} | Checkpoint: {args.checkpoint}")
        print(f"{'='*70}")
        from src.spelke_dsl import build_full_spelke_library
        from src.engine.library import Library
        from src.engine.ast_solver import ASTSolver
        import glob as _glob

        reg = build_full_spelke_library()
        lib = Library(reg)

        # Load checkpoint library if provided
        if args.checkpoint:
            checkpoint_dir = Path(args.checkpoint)
            lib_files = sorted(_glob.glob(str(checkpoint_dir / "library_cycle_*.json")))
            if lib_files:
                try:
                    lib = Library.from_checkpoint(lib_files[-1])
                    print(f"  Loaded library from {lib_files[-1]}")
                except Exception as e:
                    print(f"  Warning: could not load checkpoint ({e}), using base library")

        solver = ASTSolver(lib)
        solved = 0
        for task in tasks:
            pairs = [(ex.input.data, ex.output.data) for ex in task.train]
            for strat_name in [
                "_try_number_objects_count", "_try_number_objects_tile",
                "_try_forms_number_rotate", "_try_objects_forms_mirror_size",
                "_try_number_places_quadrant_count", "_try_forms_objects_number_count_rotate",
                "_try_agents_objects_path", "_try_objects_places_number_sorted",
                "_try_forms_objects_places", "_try_count_cells_render",
                "_try_persons_objects_reach", "_try_persons_count_agents",
                "_try_library_chains",
            ]:
                fn = getattr(solver, strat_name, None)
                if fn is None:
                    continue
                try:
                    result = fn(pairs, task.task_id)
                    if result is not None:
                        solved += 1
                        break
                except Exception:
                    pass

        rate = solved / len(tasks) if tasks else 0
        eval_results = {
            "mode": "eval_only",
            "n_tasks": len(tasks),
            "n_solved": solved,
            "solve_rate": rate,
            "checkpoint": str(args.checkpoint),
        }
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        print(f"\n  Eval-only result: {solved}/{len(tasks)} ({rate:.1%})")
        print(f"  Saved to: {output_dir}/eval_results.json")
        sys.exit(0)

    # Run Spelke experiment
    config._full_spelke = getattr(args, 'full_spelke', False)
    config._include_persons = getattr(args, 'include_persons', False)
    config._include_places = getattr(args, 'curriculum_pretrain', False)

    # Curriculum pretraining: run 1 cycle on synthetic tasks first
    if args.curriculum_pretrain:
        print(f"\n{'='*70}")
        print(f"  CURRICULUM PRETRAINING — 1 cycle on synthetic tasks")
        print(f"{'='*70}")
        from src.arc.loader import ArcDataset
        from src.arc.grid import ArcTask
        import os

        curriculum_dirs = [
            ("synth_", "data/synthetic_cross_system"),
            ("no_", "data/synthetic_number_objects"),
            ("fop_", "data/synthetic_forms_objects_places"),
            ("nor_", "data/synthetic_number_objects_replicate"),
            ("cells_", "data/synthetic_count_cells"),
            ("compound_", "data/synthetic_compounding"),
        ]
        curriculum_tasks = []
        for prefix, rel_dir in curriculum_dirs:
            d = Path(rel_dir)
            if not d.exists():
                continue
            for fname in sorted(os.listdir(d)):
                if not fname.endswith('.json'):
                    continue
                with open(d / fname) as f:
                    data = json.load(f)
                tid = prefix + fname.replace('.json', '')
                curriculum_tasks.append(ArcTask.from_dict(tid, data))

        print(f"  Loaded {len(curriculum_tasks)} curriculum tasks")

        # Run 1 cycle on curriculum tasks
        from src.engine.wake_sleep import WakeSleepConfig, WakeSleepEngine
        pretrain_config = WakeSleepConfig(
            n_iterations=1,
            search_timeout_per_task=3.0,
            max_search_attempts=500,
            enumeration_max_cost=args.enum_cost,
            enumeration_budget=args.enum_budget,
            use_enumerator=not args.no_enumerator,
            use_rust_enumerator=getattr(args, 'rust', False),
            use_neural_recognition=False,
            dream_samples_per_cycle=10,
            max_abstractions_per_cycle=15,
            verbose=args.verbose,
        )
        pretrain_config._include_places = True
        pretrain_dir = output_dir / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        pretrain_config.checkpoint_dir = str(pretrain_dir)

        pretrain_engine = WakeSleepEngine(pretrain_config)
        from src.spelke_dsl import build_spelke_library
        pretrain_reg = build_spelke_library(include_places=True)
        pretrain_engine.initialize(pretrain_reg)
        pretrain_results = pretrain_engine.run(curriculum_tasks)

        # Save pretrain checkpoint for the main run to resume from
        pretrain_data = {
            "n_curriculum_tasks": len(curriculum_tasks),
            "solved": pretrain_results[-1].n_solved if pretrain_results else 0,
            "library_size": pretrain_results[-1].total_library_size if pretrain_results else 0,
            "abstractions": len(pretrain_results[-1].new_abstractions) if pretrain_results else 0,
        }
        with open(pretrain_dir / "pretrain_results.json", "w") as f:
            json.dump(pretrain_data, f, indent=2)
        print(f"  Pretrain: {pretrain_data['solved']}/{len(curriculum_tasks)} solved, "
              f"library={pretrain_data['library_size']}")

        # Now resume main experiment from the pretrained checkpoint
        args.resume_from = str(pretrain_dir)
        args.resume_from_cycle = 0

    spelke_data = run_spelke_experiment(tasks, config, output_dir,
                                        resume_from=args.resume_from,
                                        resume_from_cycle=args.resume_from_cycle)

    def _make_baseline_config():
        return WakeSleepConfig(
            n_iterations=args.cycles,
            search_timeout_per_task=10.0,
            max_search_attempts=500,
            enumeration_max_cost=args.enum_cost,
            enumeration_budget=args.enum_budget,
            use_enumerator=not args.no_enumerator,
            use_neural_recognition=not args.no_neural,
            dream_samples_per_cycle=100 if args.quick else 200,
            max_abstractions_per_cycle=15,
            verbose=args.verbose,
        )

    # Run generic baseline if requested
    generic_data = None
    if args.run_generic:
        generic_data = run_generic_experiment(tasks, _make_baseline_config(), output_dir)

    # Run VIMRL baseline if requested
    vimrl_data = None
    if args.run_vimrl:
        vimrl_data = run_vimrl_experiment(tasks, _make_baseline_config(), output_dir)

    # Print comparison and save summary
    print_comparison(spelke_data, generic_data)
    if vimrl_data:
        s_rate = spelke_data["final_solve_rate"]
        v_rate = vimrl_data["final_solve_rate"]
        print(f"\n  Spelke vs VIMRL: {s_rate:.1%} vs {v_rate:.1%} (Δ = {s_rate - v_rate:+.1%})")
        print(f"  (VIMRL = objectness-only, no geometry/number modules)")
    save_summary(output_dir, spelke_data, generic_data, args)

    # Optional sample-efficiency curve
    if args.sample_efficiency:
        print(f"\n{'='*70}")
        print(f"  RUNNING SAMPLE-EFFICIENCY CURVE")
        print(f"{'='*70}")
        import subprocess
        se_out = output_dir / "sample_efficiency"
        se_out.mkdir(exist_ok=True)
        subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "sample_efficiency.py"),
            "--data-dir", args.data_dir,
            "--split", args.split,
            "--cycles", str(args.cycles),
            "--enum-cost", str(args.enum_cost),
            "--enum-budget", str(args.enum_budget),
            "--output-dir", str(se_out),
        ], check=False)


if __name__ == "__main__":
    main()
