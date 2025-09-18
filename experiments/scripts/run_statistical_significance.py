"""
run_statistical_significance.py — Multi-seed experiment for statistical significance.

The master plan (Section 18.2) requires:
  "Bootstrap your confidence intervals. Run multiple seeds.
   Report sample-efficiency curves, not just single solve rates."

This script:
  1. Runs the full Spelke vs Generic vs VIMRL comparison with N seeds
  2. Computes bootstrapped 95% confidence intervals for the solve-rate delta
  3. Runs McNemar's test for paired comparison on task-level outcomes
  4. Generates a JSON results file suitable for paper inclusion

Usage:
    cd bootstrap-substrate && source .venv/bin/activate
    python experiments/scripts/run_statistical_significance.py --seeds 3
    python experiments/scripts/run_statistical_significance.py --seeds 5 --quick
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_single_seed(tasks, seed, cycles, enum_cost, enum_budget, dsl_type, verbose=False, out_dir=None):
    """
    Run one seed of the full experiment. Returns per-task solve results.
    dsl_type: "spelke", "generic", or "vimrl"

    Checkpoint resume: if cycle_ckpts_<dsl>_seed<N>/ exists, resumes from
    the furthest completed full cycle, then pre-injects any mid-wake progress
    from the interrupted cycle so the wake loop skips already-solved tasks.
    """
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)

    from src.engine.wake_sleep import WakeSleepEngine, WakeSleepConfig
    from src.engine.program import Program

    cycle_ckpt_dir = None
    resume_cycle = None   # last fully completed cycle to load from
    resume_solved = set() # task IDs solved in a partial mid-wake checkpoint

    if out_dir is not None:
        cycle_ckpt_dir = str(out_dir / f"cycle_ckpts_{dsl_type}_seed{seed}")
        Path(cycle_ckpt_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(cycle_ckpt_dir)

        # Detect furthest completed full cycle checkpoint
        for c in range(cycles - 1, -1, -1):
            if (ckpt_path / f"library_cycle_{c}.json").exists():
                resume_cycle = c
                print(f"    [resume] Found full cycle checkpoint at cycle {c} for {dsl_type} seed={seed}", flush=True)
                break

        # Detect mid-wake progress from the next cycle after resume_cycle
        mid_cycle = (resume_cycle + 1) if resume_cycle is not None else 0
        mid_path = ckpt_path / f"mid_wake_cycle{mid_cycle}.json"
        if mid_path.exists():
            with open(mid_path) as f:
                mid = json.load(f)
            resume_solved = set(mid.get("solved_ids", []))
            tasks_done = mid.get("tasks_done", 0)
            print(f"    [resume] Mid-wake checkpoint: {len(resume_solved)} solved, "
                  f"{tasks_done}/{mid.get('tasks_total','?')} tasks done in cycle {mid_cycle}", flush=True)

    n_resume_cycles = cycles
    if resume_cycle is not None:
        # Only run the remaining cycles after the resumed one
        n_resume_cycles = cycles - (resume_cycle + 1)

    config = WakeSleepConfig(
        n_iterations=max(1, n_resume_cycles) if resume_cycle is not None else cycles,
        search_timeout_per_task=3.0,
        max_search_attempts=500,
        enumeration_max_cost=enum_cost,
        enumeration_budget=enum_budget,
        use_enumerator=True,
        use_neural_recognition=True,
        dream_samples_per_cycle=10,
        max_abstractions_per_cycle=15,
        verbose=verbose,
        checkpoint_dir=cycle_ckpt_dir,
    )

    engine = WakeSleepEngine(config)

    if dsl_type == "spelke":
        from src.spelke_dsl import build_spelke_library
        registry = build_spelke_library()
    elif dsl_type == "generic":
        from src.baselines.generic_dsl import build_generic_dsl
        registry = build_generic_dsl()
    elif dsl_type == "vimrl":
        from src.baselines.vimrl_dsl import build_vimrl_dsl
        registry = build_vimrl_dsl()
    else:
        raise ValueError(f"Unknown DSL type: {dsl_type}")

    engine.initialize(registry)

    # Resume from full cycle checkpoint if available
    if resume_cycle is not None:
        try:
            engine.load_checkpoint(cycle_ckpt_dir, resume_cycle)  # type: ignore[arg-type]
            print(f"    [resume] Loaded cycle {resume_cycle} checkpoint — "
                  f"{len(engine._solved_programs)} programs restored", flush=True)
        except Exception as e:
            print(f"    [resume] WARNING: checkpoint load failed ({e}), starting fresh", flush=True)

    # Pre-inject mid-wake solved IDs so wake loop skips them
    if resume_solved:
        from src.engine.program import LitNode
        injected = 0
        for tid in resume_solved:
            if tid not in engine._solved_programs:
                engine._solved_programs[tid] = Program(root=LitNode(0), task_id=tid, source="checkpoint")
                injected += 1
        if injected:
            print(f"    [resume] Pre-injected {injected} mid-wake solved tasks — wake will skip them", flush=True)

    results = engine.run(shuffled)

    # Per-task results
    solved_ids = set()
    if results:
        solved_ids = set(results[-1].solved_task_ids)

    task_results = {}
    for task in tasks:
        task_results[task.task_id] = task.task_id in solved_ids

    return {
        "seed": seed,
        "dsl": dsl_type,
        "n_tasks": len(tasks),
        "n_solved": len(solved_ids),
        "solve_rate": len(solved_ids) / len(tasks),
        "task_results": task_results,
        "n_heuristic": results[-1].n_heuristic if results else 0,
        "n_ast": results[-1].n_ast if results else 0,
        "n_enumerated": results[-1].n_enumerated if results else 0,
        "library_size": results[-1].total_library_size if results else 0,
        "cross_system": results[-1].cross_system_abstractions if results else 0,
    }


def bootstrap_ci(data, statistic_fn, n_bootstrap=10000, ci=0.95):
    """Compute bootstrapped confidence interval for a statistic."""
    rng = np.random.RandomState(42)
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        stats.append(statistic_fn(sample))
    stats = np.array(stats)
    alpha = (1 - ci) / 2
    return float(np.percentile(stats, alpha * 100)), float(np.percentile(stats, (1 - alpha) * 100))


def mcnemar_test(spelke_results, generic_results):
    """
    McNemar's test for paired nominal data.
    Compares task-level outcomes between two systems.

    Returns: (chi2, p_value, n_discordant, effect_description)
    """
    from scipy import stats as scipy_stats

    # Count discordant pairs
    b = 0  # Spelke solved, Generic didn't
    c = 0  # Generic solved, Spelke didn't

    all_tasks = set(spelke_results.keys()) & set(generic_results.keys())
    for tid in all_tasks:
        s = spelke_results[tid]
        g = generic_results[tid]
        if s and not g:
            b += 1
        elif g and not s:
            c += 1

    # McNemar's test (with continuity correction)
    if b + c == 0:
        return 0.0, 1.0, 0, "No discordant pairs"

    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
    # p-value from chi-squared distribution with 1 df
    try:
        p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)
    except Exception:
        # Fallback if scipy not available
        p_value = -1.0

    effect = f"Spelke wins {b} tasks, Generic wins {c} tasks"
    return float(chi2), float(p_value), b + c, effect


def fishers_exact_test(spelke_solved, spelke_total, generic_solved, generic_total):
    """
    Fisher's exact test comparing solve rates.
    Fallback when scipy is available.
    """
    try:
        from scipy.stats import fisher_exact
        # Contingency table: [[spelke_solved, spelke_unsolved], [generic_solved, generic_unsolved]]
        table = [
            [spelke_solved, spelke_total - spelke_solved],
            [generic_solved, generic_total - generic_solved],
        ]
        odds_ratio, p_value = fisher_exact(table)
        return float(odds_ratio), float(p_value)
    except ImportError:
        return -1.0, -1.0


def print_report(all_results, output_dir):
    """Generate and print the statistical significance report."""
    dsls = sorted(set(r["dsl"] for r in all_results))
    seeds = sorted(set(r["seed"] for r in all_results))

    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE REPORT")
    print(f"  Seeds: {len(seeds)}, DSLs: {dsls}")
    print(f"{'='*70}")

    # Per-DSL summary
    for dsl in dsls:
        dsl_runs = [r for r in all_results if r["dsl"] == dsl]
        rates = [r["solve_rate"] for r in dsl_runs]
        mean = np.mean(rates)
        std = np.std(rates)
        ci_lo, ci_hi = bootstrap_ci(np.array(rates), np.mean, n_bootstrap=5000)
        print(f"\n  {dsl.upper()}: {mean:.1%} ± {std:.1%} (95% CI: [{ci_lo:.1%}, {ci_hi:.1%}])")
        for r in dsl_runs:
            print(f"    seed {r['seed']}: {r['n_solved']}/{r['n_tasks']} ({r['solve_rate']:.1%})")

    # Pairwise comparisons
    sg_chi2, sg_p = 0.0, -1.0
    sv_chi2, sv_p = 0.0, -1.0
    if "spelke" in dsls and "generic" in dsls:
        s_runs = [r for r in all_results if r["dsl"] == "spelke"]
        g_runs = [r for r in all_results if r["dsl"] == "generic"]

        # Delta statistics
        s_rates = np.array([r["solve_rate"] for r in s_runs])
        g_rates = np.array([r["solve_rate"] for r in g_runs])

        # Paired by seed
        deltas = []
        for seed in seeds:
            sr = next((r for r in s_runs if r["seed"] == seed), None)
            gr = next((r for r in g_runs if r["seed"] == seed), None)
            if sr and gr:
                deltas.append(sr["solve_rate"] - gr["solve_rate"])

        if deltas:
            deltas = np.array(deltas)
            delta_mean = np.mean(deltas)
            delta_ci_lo, delta_ci_hi = bootstrap_ci(deltas, np.mean) if len(deltas) > 1 else (delta_mean, delta_mean)

            print(f"\n  SPELKE vs GENERIC:")
            print(f"    Δ solve rate: {delta_mean:+.1%} (95% CI: [{delta_ci_lo:+.1%}, {delta_ci_hi:+.1%}])")
            if delta_ci_lo > 0:
                print(f"    ✓ SIGNIFICANT: CI does not include 0")
            else:
                print(f"    ✗ Not significant: CI includes 0")

        # McNemar's: find first matched seed where both DSLs have task_results
        sr0, gr0 = None, None
        for s in seeds:
            sr_try = next((r for r in s_runs if r["seed"] == s), None)
            gr_try = next((r for r in g_runs if r["seed"] == s), None)
            if (sr_try and gr_try and
                    sr_try.get("task_results") and gr_try.get("task_results")):
                sr0, gr0 = sr_try, gr_try
                break
        if sr0 and gr0:
            sg_chi2, sg_p, sg_n_disc, sg_desc = mcnemar_test(sr0["task_results"], gr0["task_results"])
            print(f"    McNemar's (seed {sr0['seed']}): χ²={sg_chi2:.2f}, p={sg_p:.4f}, {sg_desc}")
            if sg_p < 0.05:
                print(f"    ✓ McNemar's significant at p < 0.05")
        else:
            print(f"    McNemar's: no seed with task_results available")

    if "spelke" in dsls and "vimrl" in dsls:
        s_runs = [r for r in all_results if r["dsl"] == "spelke"]
        v_runs = [r for r in all_results if r["dsl"] == "vimrl"]
        s_rates = np.array([r["solve_rate"] for r in s_runs])
        v_rates = np.array([r["solve_rate"] for r in v_runs])
        print(f"\n  SPELKE vs VIMRL:")
        print(f"    Spelke: {np.mean(s_rates):.1%}, VIMRL: {np.mean(v_rates):.1%}")
        print(f"    Δ = {np.mean(s_rates) - np.mean(v_rates):+.1%}")
        sr0_v, vr0 = None, None
        for s in seeds:
            sr_try = next((r for r in s_runs if r["seed"] == s), None)
            vr_try = next((r for r in v_runs if r["seed"] == s), None)
            if (sr_try and vr_try and
                    sr_try.get("task_results") and vr_try.get("task_results")):
                sr0_v, vr0 = sr_try, vr_try
                break
        if sr0_v and vr0:
            sv_chi2, sv_p, sv_n_disc, sv_desc = mcnemar_test(sr0_v["task_results"], vr0["task_results"])
            print(f"    McNemar's (seed {sr0_v['seed']}): χ²={sv_chi2:.2f}, p={sv_p:.4f}, {sv_desc}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_seeds": len(seeds),
        "dsls": dsls,
        "results": all_results,
        "summary": {},
        "spelke_vs_generic": {
            "chi2": float(sg_chi2),
            "p_value": float(sg_p),
            "n_seeds": len(seeds),
        },
        "spelke_vs_vimrl": {
            "chi2": float(sv_chi2),
            "p_value": float(sv_p),
            "n_seeds": len(seeds),
        },
    }

    for dsl in dsls:
        dsl_runs = [r for r in all_results if r["dsl"] == dsl]
        rates = [r["solve_rate"] for r in dsl_runs]
        ci_lo, ci_hi = bootstrap_ci(np.array(rates), np.mean, n_bootstrap=5000) if len(rates) > 1 else (rates[0], rates[0])
        report["summary"][dsl] = {
            "mean_solve_rate": float(np.mean(rates)),
            "std_solve_rate": float(np.std(rates)),
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "per_seed": [{"seed": r["seed"], "solve_rate": r["solve_rate"]} for r in dsl_runs],
        }

    out_path = output_dir / "statistical_significance.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/arc-agi-1/data")
    p.add_argument("--split", default="training")
    p.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--tasks", type=int, default=None)
    p.add_argument("--enum-cost", type=int, default=5)
    p.add_argument("--enum-budget", type=float, default=20.0)
    p.add_argument("--dsls", nargs="+", default=["spelke", "generic", "vimrl"],
                   help="Which DSLs to run")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 50 tasks, 2 cycles, cost=4")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.tasks = args.tasks or 50
        args.cycles = 2
        args.enum_cost = 4
        args.enum_budget = 10.0
        print("  [Quick mode: 50 tasks, 2 cycles, cost=4]")

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quick" if args.quick else "full"
        out_dir = Path("experiments/outputs") / f"stat_significance_{mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    print(f"\nLoading ARC tasks from {args.data_dir}/{args.split}...")
    from src.arc.loader import ArcDataset
    dataset = ArcDataset(args.data_dir)
    tasks = dataset.load_split(args.split)
    if args.tasks:
        tasks = tasks[:args.tasks]
    print(f"Loaded {len(tasks)} tasks")

    print(f"\nConfig: seeds={args.seeds}, cycles={args.cycles}, dsls={args.dsls}")
    print(f"        enum_cost={args.enum_cost}, enum_budget={args.enum_budget}s")
    print(f"        Output: {out_dir}")

    all_results = []
    total_runs = args.seeds * len(args.dsls)
    done = 0

    # Resume from checkpoint if it exists
    completed = set()
    ckpt_path = out_dir / "checkpoint.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_results = ckpt.get("results_so_far", [])
        done = ckpt.get("done", 0)
        completed = {(r["seed"], r["dsl"]) for r in all_results}
        print(f"\nResuming from checkpoint: {done}/{total_runs} runs already done")
        for r in all_results:
            print(f"  skipping {r['dsl']} seed={r['seed']} (already done: {r['n_solved']}/400)")

    for seed in range(args.seeds):
        for dsl in args.dsls:
            if (seed, dsl) in completed:
                continue
            done += 1
            print(f"\n  [{done}/{total_runs}] {dsl.upper()} seed={seed} ...", flush=True)
            t0 = time.time()
            result = run_single_seed(
                tasks, seed, args.cycles, args.enum_cost, args.enum_budget,
                dsl, verbose=args.verbose, out_dir=out_dir,
            )
            elapsed = time.time() - t0
            print(f"    → {result['n_solved']}/{result['n_tasks']} ({result['solve_rate']:.1%}) "
                  f"in {elapsed:.0f}s")

            # Remove task_results from serialization (too large)
            serializable = {k: v for k, v in result.items() if k != "task_results"}
            all_results.append(result)

            # Checkpoint after each run
            ckpt = {
                "results_so_far": [
                    {k: v for k, v in r.items() if k != "task_results"}
                    for r in all_results
                ],
                "done": done,
                "total": total_runs,
            }
            with open(out_dir / "checkpoint.json", "w") as f:
                json.dump(ckpt, f, indent=2)

    print_report(all_results, out_dir)


if __name__ == "__main__":
    main()
