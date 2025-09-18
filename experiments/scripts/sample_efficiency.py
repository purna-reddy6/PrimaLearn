"""
sample_efficiency.py — Solve rate vs. training task count experiment.

The master plan (Section 11.5) requires a sample-efficiency curve:
  solve rate as a function of training task count (10, 50, 100, 200, 400)

This is the paper's most critical figure. If Spelke-initialized DreamCoder
pulls ahead of the generic DSL at low task counts, that IS the sample-
efficiency story that the hypothesis predicts.

Each data point runs the full 3-cycle wake-sleep loop on a random subset
of ARC-AGI-1 tasks, then evaluates on those same tasks (in-distribution).
Repeated with 3 random seeds for error bars.

Usage:
    cd bootstrap-substrate
    source .venv/bin/activate
    python experiments/scripts/sample_efficiency.py [--quick] [--seeds 3]
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


TASK_COUNTS = [10, 25, 50, 100, 200, 400]
DEFAULT_SEEDS = 3
DEFAULT_CYCLES = 3
DEFAULT_ENUM_COST = 5   # Higher than main experiment — this is where Carey signal lives
DEFAULT_ENUM_BUDGET = 20.0


def run_one(tasks, n_tasks, seed, cycles, enum_cost, enum_budget, use_spelke, verbose=False):
    """Run a single (n_tasks, seed) data point. Returns solve_rate per cycle."""
    import numpy as np
    rng = random.Random(seed)
    subset = rng.sample(tasks, min(n_tasks, len(tasks)))

    from src.engine.wake_sleep import WakeSleepEngine, WakeSleepConfig

    config = WakeSleepConfig(
        n_iterations=cycles,
        search_timeout_per_task=5.0,
        max_search_attempts=300,
        enumeration_max_cost=enum_cost,
        enumeration_budget=enum_budget,
        use_enumerator=True,
        use_neural_recognition=True,
        dream_samples_per_cycle=5,
        max_abstractions_per_cycle=5,
        verbose=verbose,
    )

    engine = WakeSleepEngine(config)

    if use_spelke:
        from src.spelke_dsl import build_spelke_library
        registry = build_spelke_library()
    else:
        from src.baselines.generic_dsl import build_generic_dsl
        registry = build_generic_dsl()

    engine.initialize(registry)
    results = engine.run(subset)

    return {
        "n_tasks": n_tasks,
        "seed": seed,
        "system": "spelke" if use_spelke else "generic",
        "solve_rates": [r.solve_rate for r in results],
        "final_solve_rate": results[-1].solve_rate if results else 0.0,
        "final_n_solved": results[-1].n_solved if results else 0,
        "n_heuristic": results[-1].n_heuristic if results else 0,
        "n_ast": results[-1].n_ast if results else 0,
        "n_enumerated": results[-1].n_enumerated if results else 0,
        "library_size": results[-1].total_library_size if results else 0,
        "cross_system_abstractions": results[-1].cross_system_abstractions if results else 0,
    }


def _ckpt_key(label, n, seed):
    return f"{label.lower()}_n{n}_s{seed}"


def run_curve(tasks, task_counts, seeds, cycles, enum_cost, enum_budget, use_spelke, label,
              checkpoint_path=None):
    """Run the full sample-efficiency curve, with optional checkpoint/resume."""
    import numpy as np

    # Load existing checkpoint
    ckpt = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        print(f"  [checkpoint] Resuming from {checkpoint_path} ({len(ckpt)} cached runs)")

    results = []
    total = len(task_counts) * seeds
    done = 0

    for n in task_counts:
        seed_results = []
        for seed in range(seeds):
            done += 1
            key = _ckpt_key(label, n, seed)
            if key in ckpt:
                r = ckpt[key]
                print(f"  [{done}/{total}] {label} | n={n} seed={seed} ... "
                      f"solve={r['final_solve_rate']:.1%} (cached)")
                seed_results.append(r)
                continue

            print(f"  [{done}/{total}] {label} | n={n} seed={seed} ... ", end="", flush=True)
            t0 = time.time()
            r = run_one(tasks, n, seed, cycles, enum_cost, enum_budget, use_spelke)
            elapsed = time.time() - t0
            print(f"solve={r['final_solve_rate']:.1%} ({elapsed:.0f}s)")
            seed_results.append(r)

            # Save incremental checkpoint
            if checkpoint_path:
                ckpt[key] = r
                with open(checkpoint_path, "w") as f:
                    json.dump(ckpt, f, indent=2)

        # Aggregate across seeds
        rates = [r["final_solve_rate"] for r in seed_results]
        results.append({
            "n_tasks": n,
            "mean_solve_rate": float(np.mean(rates)),
            "std_solve_rate": float(np.std(rates)),
            "min_solve_rate": float(np.min(rates)),
            "max_solve_rate": float(np.max(rates)),
            "seeds": seed_results,
        })

    return results


def print_table(spelke_curve, generic_curve):
    print("\n" + "="*70)
    print("  SAMPLE EFFICIENCY RESULTS")
    print("="*70)
    print(f"\n  {'N Tasks':<10} {'Spelke':<15} {'Generic':<15} {'Delta':<10}")
    print(f"  {'-'*50}")

    for s, g in zip(spelke_curve, generic_curve):
        n = s["n_tasks"]
        s_rate = s["mean_solve_rate"]
        s_std = s["std_solve_rate"]
        g_rate = g["mean_solve_rate"]
        g_std = g["std_solve_rate"]
        delta = s_rate - g_rate
        flag = " ✓" if delta > 0.01 else ""
        print(f"  {n:<10} {s_rate:.1%}±{s_std:.1%}  {g_rate:.1%}±{g_std:.1%}  {delta:+.1%}{flag}")

    # Find crossover point
    for s, g in zip(spelke_curve, generic_curve):
        if s["mean_solve_rate"] > g["mean_solve_rate"] + 0.005:
            print(f"\n  Spelke advantage appears at n={s['n_tasks']} tasks")
            print(f"  This is the sample-efficiency crossover predicted by the hypothesis.")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/arc-agi-1/data")
    p.add_argument("--split", default="training")
    p.add_argument("--task-counts", nargs="+", default=None,
                   help="Task counts to evaluate, space- or comma-separated "
                        "(default: 10 25 50 100 200 400)")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    p.add_argument("--enum-cost", type=int, default=DEFAULT_ENUM_COST)
    p.add_argument("--enum-budget", type=float, default=DEFAULT_ENUM_BUDGET)
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: counts=[10,25,50], 1 seed, 2 cycles, cost=4")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.quick:
        task_counts = [10, 25, 50]
        seeds = 1
        cycles = 2
        enum_cost = 4
        enum_budget = 10.0
        print("  [Quick mode: n=[10,25,50], 1 seed, 2 cycles, cost=4]")
    else:
        if args.task_counts:
            # Support both space-separated and comma-separated: "10,25,50" or "10 25 50"
            raw = []
            for item in args.task_counts:
                raw.extend(item.split(","))
            task_counts = [int(x) for x in raw if x.strip()]
        else:
            task_counts = TASK_COUNTS
        seeds = args.seeds
        cycles = args.cycles
        enum_cost = args.enum_cost
        enum_budget = args.enum_budget

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quick" if args.quick else "full"
        out_dir = Path("experiments/outputs") / f"sample_efficiency_{mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    print(f"\nLoading ARC tasks from {args.data_dir}/{args.split}...")
    from src.arc.loader import ArcDataset
    dataset = ArcDataset(args.data_dir)
    tasks = dataset.load_split(args.split)
    print(f"Loaded {len(tasks)} tasks")

    print(f"\nConfig: task_counts={task_counts}, seeds={seeds}, cycles={cycles}")
    print(f"        enum_cost={enum_cost}, enum_budget={enum_budget}s")
    print(f"        Output: {out_dir}")

    ckpt_path = str(out_dir / "curve_checkpoint.json")

    # Run Spelke curve
    print(f"\n{'='*70}")
    print(f"  SPELKE-INITIALIZED DREAMCODER — sample efficiency curve")
    print(f"{'='*70}")
    t0 = time.time()
    spelke_curve = run_curve(
        tasks, task_counts, seeds, cycles, enum_cost, enum_budget,
        use_spelke=True, label="Spelke", checkpoint_path=ckpt_path,
    )
    spelke_time = time.time() - t0

    # Run Generic curve
    print(f"\n{'='*70}")
    print(f"  GENERIC DSL BASELINE — sample efficiency curve")
    print(f"{'='*70}")
    t0 = time.time()
    generic_curve = run_curve(
        tasks, task_counts, seeds, cycles, enum_cost, enum_budget,
        use_spelke=False, label="Generic", checkpoint_path=ckpt_path,
    )
    generic_time = time.time() - t0

    print_table(spelke_curve, generic_curve)

    # Save
    results = {
        "config": {
            "task_counts": task_counts,
            "seeds": seeds,
            "cycles": cycles,
            "enum_cost": enum_cost,
            "enum_budget": enum_budget,
        },
        "spelke": spelke_curve,
        "generic": generic_curve,
        "total_time_seconds": spelke_time + generic_time,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = out_dir / "sample_efficiency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save a simple CSV for plotting
    csv_path = out_dir / "sample_efficiency.csv"
    with open(csv_path, "w") as f:
        f.write("n_tasks,spelke_mean,spelke_std,generic_mean,generic_std,delta\n")
        for s, g in zip(spelke_curve, generic_curve):
            delta = s["mean_solve_rate"] - g["mean_solve_rate"]
            f.write(f"{s['n_tasks']},{s['mean_solve_rate']:.4f},{s['std_solve_rate']:.4f},"
                    f"{g['mean_solve_rate']:.4f},{g['std_solve_rate']:.4f},{delta:.4f}\n")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
