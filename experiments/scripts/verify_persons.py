#!/usr/bin/env python3
"""
verify_persons.py — Phase 2 PERSONS milestone gate (10 checks).

P1:  src/spelke_dsl/l_persons.py exists with ≥10 primitives registered
P2:  SpelkeSystem.PERSONS exists in base.py enum
P3:  data/synthetic_persons_objects/ has ≥30 tasks
P4:  AST solver solves ≥25/30 PERSONS tasks
P5:  Generic DSL solves 0/30 PERSONS tasks (Spelke-specific tasks confirmed)
P6:  curriculum demo on synthetic_persons_objects finds ≥1 is_cross_system
     abstraction with PERSONS in systems_composed
P7:  Full 6-system experiment results exist in experiments/outputs/six_system_run/
P8:  verify_phase1.py still 10/10 (no regressions)
P9:  211+ tests pass (124 original + 87 PERSONS)
P10: Paper contains "PERSONS" and "six" (or "6") Spelke systems

Prints: PERSONS COMPLETE N/10
Exits 1 if any check fails.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # bootstrap-substrate/
PERSONS_TASKS_DIR = ROOT / "data/synthetic_persons_objects"
SIX_SYS_DIR = ROOT / "experiments/outputs/six_system_run"
DEMO_RESULTS = ROOT / "experiments/outputs/persons_curriculum_demo/curriculum_demo_results.json"
PAPER_PATH = ROOT / "paper/paper.html"

PASS_COUNT = 0
ALL_CHECKS = []


def check(label: str, passed: bool, detail: str = "") -> bool:
    global PASS_COUNT
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if passed:
        PASS_COUNT += 1
    ALL_CHECKS.append((label, passed, detail))
    return passed


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def run_tests() -> tuple[bool, int]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=ROOT
    )
    for line in result.stdout.splitlines():
        if "passed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed":
                    try:
                        count = int(parts[i - 1])
                        return (count >= 211, count)
                    except (ValueError, IndexError):
                        pass
    return (False, 0)


def count_persons_primitives() -> int:
    """Count PERSONS primitives registered in the library."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from src.spelke_dsl.l_persons import register_persons_primitives
        from src.spelke_dsl.base import PrimitiveRegistry
        reg = PrimitiveRegistry()
        register_persons_primitives(reg)
        return len(reg)
    except Exception as e:
        print(f"       Error importing l_persons: {e}")
        return 0


def run_ast_solver_on_persons_tasks(use_spelke: bool = True) -> tuple[int, int]:
    """
    Run AST solver on synthetic_persons_objects tasks.
    Returns (solved_count, total_count).
    use_spelke=True: include PERSONS primitives
    use_spelke=False: generic DSL (no PERSONS)
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        import numpy as np
        import json as _json

        from src.spelke_dsl import build_spelke_library
        from src.engine.library import Library
        from src.engine.ast_solver import ASTSolver
        from src.arc.grid import ArcTask, TaskExample, Grid

        if use_spelke:
            reg = build_spelke_library(include_persons=True)
        else:
            reg = build_spelke_library()  # no PERSONS

        lib = Library(reg)
        solver = ASTSolver(lib)

        task_files = sorted(PERSONS_TASKS_DIR.glob("*.json"))
        solved = 0
        total = 0
        for tf in task_files:
            try:
                data = _json.loads(tf.read_text())
                train_examples = [TaskExample.from_dict(ex) for ex in data.get("train", [])]
                if not train_examples:
                    continue
                task = ArcTask(
                    task_id=data.get("task_id", tf.stem),
                    train=train_examples,
                    test=train_examples[:1],
                )
                total += 1
                prog = solver.solve(task)
                if prog is not None:
                    solved += 1
            except Exception:
                total += 1  # count as attempted but failed
        return (solved, total)
    except Exception as e:
        print(f"       AST solver error: {e}")
        return (0, 0)


def check_p6_curriculum_demo() -> tuple[bool, str]:
    """Check if curriculum demo found ≥1 is_cross_system abstraction with PERSONS."""
    demo = load_json(DEMO_RESULTS)
    if not demo:
        return False, "demo results not found (run T7 first)"

    cross_system_with_persons = []
    for cycle_key, cycle_data in demo.get("spelke", {}).items():
        for ab in cycle_data.get("abstraction_details", []):
            if ab.get("is_cross_system", False):
                systems = ab.get("systems", [])
                if any(s.upper() == "PERSONS" for s in systems):
                    cross_system_with_persons.append(ab)

    if cross_system_with_persons:
        ab = cross_system_with_persons[0]
        return True, (f"{len(cross_system_with_persons)} found — "
                      f"e.g. {ab['name']}: systems={ab.get('systems', [])}")
    else:
        all_cross = []
        for cycle_key, cycle_data in demo.get("spelke", {}).items():
            for ab in cycle_data.get("abstraction_details", []):
                if ab.get("is_cross_system", False):
                    all_cross.append(ab)
        return False, (f"0 with PERSONS in systems_composed "
                       f"(total cross-system abs found: {len(all_cross)})")


def check_p7_six_system_run() -> tuple[bool, str]:
    """Check if six_system_run results exist."""
    if not SIX_SYS_DIR.exists():
        return False, f"directory not found: {SIX_SYS_DIR}"

    # Check for any results file
    for fname in ["results.json", "summary.json"]:
        for subdir in [SIX_SYS_DIR, SIX_SYS_DIR / "spelke"]:
            candidate = subdir / fname
            if candidate.exists():
                data = load_json(candidate)
                solved = data.get("arc_solved", data.get("solved", "?"))
                return True, f"{candidate.relative_to(ROOT)} — solved={solved}"

    return False, "no results.json found in six_system_run/"


def check_p8_phase1() -> tuple[bool, str]:
    """Run verify_phase1.py and check for 10/10."""
    phase1_script = ROOT / "experiments/scripts/verify_phase1.py"
    if not phase1_script.exists():
        return False, "verify_phase1.py not found"

    result = subprocess.run(
        [sys.executable, str(phase1_script)],
        capture_output=True, text=True, cwd=ROOT
    )
    output = result.stdout + result.stderr
    if "10/10" in output:
        return True, "10/10"
    # Try to extract the score
    for line in output.splitlines():
        if "/" in line and ("PASS" in line or "pass" in line or "Phase" in line.title()):
            return (result.returncode == 0), line.strip()
    return result.returncode == 0, f"exit={result.returncode}"


def check_p10_paper() -> tuple[bool, str]:
    """Check paper contains PERSONS and six/6 Spelke systems."""
    if not PAPER_PATH.exists():
        return False, "paper.html not found"

    content = PAPER_PATH.read_text(errors="ignore").lower()
    has_persons = "persons" in content
    has_six = "six" in content or " 6 " in content or ">6<" in content
    if has_persons and has_six:
        return True, "paper contains 'PERSONS' and 'six/6' Spelke systems"
    missing = []
    if not has_persons:
        missing.append("'PERSONS' not found")
    if not has_six:
        missing.append("'six/6' not found")
    return False, "; ".join(missing)


def main():
    print("=" * 65)
    print("verify_persons.py — Phase 2 PERSONS Milestone Gate")
    print("=" * 65)

    # ── P1: l_persons.py exists with ≥10 primitives ──────────────
    print("\nP1: l_persons.py exists with ≥10 primitives")
    l_persons_path = ROOT / "src/spelke_dsl/l_persons.py"
    if l_persons_path.exists():
        prim_count = count_persons_primitives()
        check("P1: l_persons.py with ≥10 PERSONS primitives",
              prim_count >= 10,
              f"{prim_count} primitives registered")
    else:
        check("P1: l_persons.py with ≥10 PERSONS primitives",
              False, "file not found")

    # ── P2: SpelkeSystem.PERSONS in enum ─────────────────────────
    print("\nP2: SpelkeSystem.PERSONS in base.py enum")
    try:
        from src.spelke_dsl.base import SpelkeSystem
        has_persons = hasattr(SpelkeSystem, "PERSONS")
        check("P2: SpelkeSystem.PERSONS in enum",
              has_persons,
              f"{'found' if has_persons else 'NOT found'}")
    except Exception as e:
        check("P2: SpelkeSystem.PERSONS in enum", False, str(e))

    # ── P3: ≥30 tasks in synthetic_persons_objects/ ───────────────
    print("\nP3: ≥30 tasks in data/synthetic_persons_objects/")
    task_files = list(PERSONS_TASKS_DIR.glob("*.json")) if PERSONS_TASKS_DIR.exists() else []
    check("P3: ≥30 tasks in synthetic_persons_objects/",
          len(task_files) >= 30,
          f"{len(task_files)} tasks found")

    # ── P4: AST solver solves ≥25/30 ─────────────────────────────
    print("\nP4: AST solver solves ≥25/30 PERSONS tasks (Spelke library)")
    if len(task_files) >= 30:
        solved, total = run_ast_solver_on_persons_tasks(use_spelke=True)
        check("P4: AST solver ≥25/30 PERSONS tasks",
              solved >= 25,
              f"{solved}/{total} solved")
    else:
        check("P4: AST solver ≥25/30 PERSONS tasks",
              False,
              "skipped — need ≥30 tasks first (P3 failed)")

    # ── P5: Generic DSL solves 0/30 ──────────────────────────────
    print("\nP5: Generic DSL (no PERSONS) solves 0/30 PERSONS tasks")
    if len(task_files) >= 30:
        solved_generic, total_generic = run_ast_solver_on_persons_tasks(use_spelke=False)
        check("P5: Generic DSL solves 0/30 (Spelke-specific tasks confirmed)",
              solved_generic == 0,
              f"{solved_generic}/{total_generic} solved by generic DSL")
    else:
        check("P5: Generic DSL solves 0/30",
              False,
              "skipped — need ≥30 tasks first (P3 failed)")

    # ── P6: Curriculum demo finds ≥1 PERSONS cross-system abs ────
    print("\nP6: Curriculum demo finds ≥1 PERSONS cross-system abstraction")
    p6_pass, p6_detail = check_p6_curriculum_demo()
    check("P6: ≥1 is_cross_system abstraction with PERSONS in systems_composed",
          p6_pass, p6_detail)

    # ── P7: Six-system run results exist ─────────────────────────
    print("\nP7: Full 6-system experiment results exist")
    p7_pass, p7_detail = check_p7_six_system_run()
    check("P7: experiments/outputs/six_system_run/ with results",
          p7_pass, p7_detail)

    # ── P8: verify_phase1.py still 10/10 ─────────────────────────
    print("\nP8: verify_phase1.py still 10/10 (no regressions)")
    p8_pass, p8_detail = check_p8_phase1()
    check("P8: verify_phase1.py 10/10", p8_pass, p8_detail)

    # ── P9: 211+ tests pass ───────────────────────────────────────
    print("\nP9: 211+ tests pass (124 original + 87 PERSONS)")
    p9_pass, test_count = run_tests()
    check("P9: 211+ tests pass", p9_pass, f"{test_count} tests pass")

    # ── P10: Paper contains PERSONS + six systems ─────────────────
    print("\nP10: Paper contains 'PERSONS' and six Spelke systems")
    p10_pass, p10_detail = check_p10_paper()
    check("P10: paper contains PERSONS + six systems", p10_pass, p10_detail)

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"PERSONS COMPLETE {PASS_COUNT}/10")
    print("=" * 65)

    if PASS_COUNT < 10:
        print("\nFailing checks:")
        for label, passed, detail in ALL_CHECKS:
            if not passed:
                print(f"  → {label}: {detail}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
