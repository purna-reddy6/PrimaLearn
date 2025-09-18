"""
verify_benchmark_v2.py — Verify benchmark v2 structure, Tier 3 properties, and compounding.

10 checks:
  V1:  data/spelke_benchmark_v2/train/ has ≥875 tasks, test/ has exactly 200
  V2:  manifest.json has ≥75 tasks labeled tier3_*
  V3:  AST solver (no checkpoint) solves 0/20 random Tier 3 tasks
  V4:  Manually verified: rotate90(abs_0_3/4/5(input)) solves T3-A/B/C tasks
  V5:  benchmark_v2_run/ exists with ≥2 completed cycles
  V6:  Cycle 0 solve rate < 95%  (Tier 3 tasks are hard)
  V7:  Cycle 1 solve rate > cycle 0 solve rate  ← THE KEY COMPOUNDING SIGNAL
  V8:  Library grows from cycle 0 to cycle 1
  V9:  Test set eval results exist in benchmark_test_eval/
  V10: 220+ tests pass

Usage:
    cd bootstrap-substrate
    python3 experiments/scripts/verify_benchmark_v2.py
"""

from __future__ import annotations
import json
import os
import sys
import glob
import random
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
TRAIN_DIR = os.path.join(ROOT, 'data', 'spelke_benchmark_v2', 'train')
TEST_DIR  = os.path.join(ROOT, 'data', 'spelke_benchmark_v2', 'test')
MANIFEST  = os.path.join(ROOT, 'data', 'spelke_benchmark_v2', 'manifest.json')
V2_RUN    = os.path.join(ROOT, 'experiments', 'outputs', 'benchmark_v2_run')
TEST_EVAL = os.path.join(ROOT, 'experiments', 'outputs', 'benchmark_test_eval')

passed = 0
failed = 0


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))
    return ok


# ── V1: Dataset structure ────────────────────────────────────────────────────
print("V1: Dataset structure")
train_files = glob.glob(os.path.join(TRAIN_DIR, '*.json'))
test_files  = glob.glob(os.path.join(TEST_DIR,  '*.json'))
n_train = len(train_files)
n_test  = len(test_files)
check("V1: train ≥875, test == 200",
      n_train >= 875 and n_test == 200,
      f"train={n_train}, test={n_test}")

# ── V2: Manifest has ≥75 Tier 3 tasks ───────────────────────────────────────
print("\nV2: Manifest Tier 3 count")
try:
    with open(MANIFEST) as f:
        manifest = json.load(f)
    tier3_tasks = [k for k, v in manifest['tasks'].items()
                   if isinstance(v, dict) and str(v.get('type', '')).startswith('tier3_')]
    check("V2: manifest has ≥75 tier3_* tasks", len(tier3_tasks) >= 75,
          f"{len(tier3_tasks)} tier3 tasks found")
except Exception as e:
    check("V2: manifest.json readable", False, str(e))
    tier3_tasks = []

# ── V3: AST solver (no checkpoint) solves 0/20 Tier 3 tasks ─────────────────
print("\nV3: AST solver (base library) solves 0/20 Tier 3")
try:
    from src.spelke_dsl import build_full_spelke_library
    from src.engine.library import Library
    from src.engine.ast_solver import ASTSolver
    from src.arc.grid import ArcTask

    reg = build_full_spelke_library()
    lib = Library(reg)
    solver = ASTSolver(lib)

    # Pick 20 random Tier 3 tasks (across all 3 types)
    tier3_files = [os.path.join(TRAIN_DIR, k + '.json') for k in tier3_tasks]
    tier3_files = [f for f in tier3_files if os.path.exists(f)]
    sample = random.sample(tier3_files, min(20, len(tier3_files)))

    v3_solved = 0
    for fname in sample:
        with open(fname) as f:
            data = json.load(f)
        task = ArcTask.from_dict(os.path.basename(fname).replace('.json', ''), data)
        result = solver.solve(task)
        if result is not None:
            v3_solved += 1

    check("V3: AST (no checkpoint) solves 0/20 Tier 3", v3_solved == 0,
          f"{v3_solved}/20 solved (need 0)")
except Exception as e:
    check("V3: AST solver check", False, f"error: {e}")

# ── V4: Verified: rotate90(abs_0_X(input)) solves Tier 3 ────────────────────
print("\nV4: Manually verify rotate90(abs_0_X) solves T3 patterns")
try:
    from src.spelke_dsl.l_objects import _extract_objects
    from src.spelke_dsl.l_number import _render_count_colored
    from src.spelke_dsl.l_forms import _flip_horizontal

    # Manually implement the known abstractions
    def abs_0_3_impl(grid: np.ndarray) -> np.ndarray:
        objs = _extract_objects(grid)
        if not objs:
            return grid
        largest = max(objs, key=lambda o: o.size)
        count = len(objs)
        color = int(largest.color)
        return _render_count_colored(count, color)

    def abs_0_4_impl(grid: np.ndarray) -> np.ndarray:
        objs = _extract_objects(grid)
        if not objs:
            return grid
        largest = max(objs, key=lambda o: o.size)
        n_tiles = len(objs)
        rendered = largest.to_grid()
        return np.hstack([rendered] * n_tiles).astype(int)

    def abs_0_5_impl(grid: np.ndarray) -> np.ndarray:
        objs = _extract_objects(grid)
        if not objs:
            return grid
        largest = max(objs, key=lambda o: o.size)
        rendered = largest.to_grid()
        return _flip_horizontal(rendered)

    # T3-A: rotate90(abs_0_3(input))
    tier3a_files = glob.glob(os.path.join(TRAIN_DIR, 'tier3a_*.json'))
    a_solved = 0
    for fname in tier3a_files[:10]:
        with open(fname) as f:
            task = json.load(f)
        ok = True
        for ex in task['train']:
            inp = np.array(ex['input'])
            expected = np.array(ex['output'])
            try:
                out = np.rot90(abs_0_3_impl(inp), k=-1).copy()
                if not np.array_equal(out, expected):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            a_solved += 1

    # T3-B: rotate90(abs_0_4(input))
    tier3b_files = glob.glob(os.path.join(TRAIN_DIR, 'tier3b_*.json'))
    b_solved = 0
    for fname in tier3b_files[:10]:
        with open(fname) as f:
            task = json.load(f)
        ok = True
        for ex in task['train']:
            inp = np.array(ex['input'])
            expected = np.array(ex['output'])
            try:
                out = np.rot90(abs_0_4_impl(inp), k=-1).copy()
                if not np.array_equal(out, expected):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            b_solved += 1

    # T3-C: rotate90(abs_0_5(input))
    tier3c_files = glob.glob(os.path.join(TRAIN_DIR, 'tier3c_*.json'))
    c_solved = 0
    for fname in tier3c_files[:10]:
        with open(fname) as f:
            task = json.load(f)
        ok = True
        for ex in task['train']:
            inp = np.array(ex['input'])
            expected = np.array(ex['output'])
            try:
                out = np.rot90(abs_0_5_impl(inp), k=-1).copy()
                if not np.array_equal(out, expected):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            c_solved += 1

    total_v4 = a_solved + b_solved + c_solved
    check("V4: rotate90(abs_0_3/4/5) solves ≥24/30 sample T3 tasks",
          a_solved + b_solved + c_solved >= 24,
          f"T3-A={a_solved}/10, T3-B={b_solved}/10, T3-C={c_solved}/10")
except Exception as e:
    check("V4: abstraction verification", False, f"error: {e}")

# ── V5: benchmark_v2_run exists with ≥2 cycles ──────────────────────────────
print("\nV5: benchmark_v2_run experiment results")
spelke_dir = os.path.join(V2_RUN, 'spelke')
cycle_files = glob.glob(os.path.join(spelke_dir, 'cycle_results.json'))
cycle_result_exists = bool(cycle_files)
n_cycles = 0
cycle_results = []
if cycle_result_exists:
    try:
        with open(cycle_files[0]) as f:
            cycle_results = json.load(f)
        n_cycles = len(cycle_results)
    except Exception:
        pass
check("V5: benchmark_v2_run/ with ≥2 cycles", n_cycles >= 2,
      f"{n_cycles} cycles found" if n_cycles > 0 else "No cycle results found")

# ── V6: Cycle 0 solve rate < 95% ────────────────────────────────────────────
print("\nV6: Cycle 0 solve rate < 95%")
c0_rate = None
c1_rate = None
if cycle_results:
    c0 = next((c for c in cycle_results if c.get('cycle') == 0), None)
    c1 = next((c for c in cycle_results if c.get('cycle') == 1), None)
    if c0:
        c0_rate = c0.get('solve_rate', None)
    if c1:
        c1_rate = c1.get('solve_rate', None)
check("V6: cycle 0 rate < 95%",
      c0_rate is not None and c0_rate < 0.95,
      f"cycle 0 = {c0_rate:.1%}" if c0_rate is not None else "no data")

# ── V7: Cycle 1 rate > cycle 0 rate ─────────────────────────────────────────
print("\nV7: Cycle 1 solve rate > cycle 0 (COMPOUNDING SIGNAL)")
check("V7: cycle 1 > cycle 0 (compounding!)",
      c0_rate is not None and c1_rate is not None and c1_rate > c0_rate,
      f"cycle 0={c0_rate:.1%} → cycle 1={c1_rate:.1%}" if (c0_rate and c1_rate) else "no data")

# ── V8: Library grows from cycle 0 to cycle 1 ────────────────────────────────
print("\nV8: Library grows")
lib0_size = None
lib1_size = None
lib0_path = os.path.join(spelke_dir, 'library_cycle_0.json')
lib1_path = os.path.join(spelke_dir, 'library_cycle_1.json')
if os.path.exists(lib0_path) and os.path.exists(lib1_path):
    try:
        with open(lib0_path) as f:
            lib0 = json.load(f)
        with open(lib1_path) as f:
            lib1 = json.load(f)
        lib0_size = len(lib0.get('primitives', []) or
                        lib0.get('base_registry', {}).get('primitives', []))
        lib1_size = len(lib1.get('primitives', []) or
                        lib1.get('base_registry', {}).get('primitives', []))
    except Exception as e:
        pass
if lib0_size is None and cycle_results:
    c0 = next((c for c in cycle_results if c.get('cycle') == 0), None)
    c1 = next((c for c in cycle_results if c.get('cycle') == 1), None)
    if c0:
        lib0_size = c0.get('total_library_size', None)
    if c1:
        lib1_size = c1.get('total_library_size', None)
check("V8: library grows cycle 0 → 1",
      lib0_size is not None and lib1_size is not None and lib1_size > lib0_size,
      f"cycle 0={lib0_size} → cycle 1={lib1_size}" if (lib0_size and lib1_size) else "no data")

# ── V9: Test set eval exists ─────────────────────────────────────────────────
print("\nV9: Test set eval results")
eval_file = os.path.join(TEST_EVAL, 'eval_results.json')
if os.path.exists(eval_file):
    try:
        with open(eval_file) as f:
            eval_res = json.load(f)
        check("V9: test eval exists",
              True,
              f"{eval_res.get('n_solved')}/{eval_res.get('n_tasks')} ({eval_res.get('solve_rate', 0):.1%})")
    except Exception as e:
        check("V9: test eval readable", False, str(e))
else:
    check("V9: benchmark_test_eval/eval_results.json exists", False, "file not found")

# ── V10: 220+ tests pass ─────────────────────────────────────────────────────
print("\nV10: Test suite")
try:
    result = subprocess.run(
        ['python3', '-m', 'pytest', 'tests/', '-q', '--tb=no'],
        capture_output=True, text=True,
        cwd=ROOT, timeout=120
    )
    output = result.stdout + result.stderr
    # Parse: "220 passed" or similar
    import re
    m = re.search(r'(\d+) passed', output)
    n_passing = int(m.group(1)) if m else 0
    check("V10: ≥220 tests pass", n_passing >= 220, f"{n_passing} tests passing")
except Exception as e:
    check("V10: pytest", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BENCHMARK V2 COMPLETE {passed}/10")
print(f"  Passed: {passed}   Failed: {failed}")
if failed > 0:
    print("\nTo complete:")
    if not os.path.exists(os.path.join(spelke_dir, 'cycle_results.json')):
        print("  Run: python3 experiments/scripts/run_experiment.py \\")
        print("    --task-dir data/spelke_benchmark_v2/train \\")
        print("    --cycles 5 --enum-cost 6 --enum-budget 15 --rust \\")
        print("    --full-spelke --output-dir experiments/outputs/benchmark_v2_run --verbose")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
