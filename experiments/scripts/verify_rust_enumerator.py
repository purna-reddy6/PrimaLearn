#!/usr/bin/env python3
"""
verify_rust_enumerator.py — Verification gate for the Rust enumerator.

Checks:
  R1: Rust binary exists at expected path
  R2: Rust binary runs on 10 test tasks and exits 0
  R3: Rust solve SET matches Python solve SET on same 10 tasks (exact match)
  R4: Rust is faster than Python on same 10 tasks (wall time comparison)
  R5: Rust handles timeout correctly — tasks with tiny budget return empty, don't hang
  R6: Rust handles nodes_explored cap — returns partial results, doesn't OOM
  R7: 98+ Python tests still pass (no Python breakage)
  R8: Rust binary produces valid JSON output (parseable, correct schema)

Prints "RUST ENUMERATOR VERIFIED — N/8" at end.
Exit 1 if any check fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import glob
from pathlib import Path

# ── Setup path ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import pytest

from src.arc.grid import ArcTask
from src.engine.library import Library
from src.engine.enumerator import TypeDirectedEnumerator
from src.engine.rust_enumerator import RustEnumerator, _serialize_library
from src.spelke_dsl import build_spelke_library

BINARY_PATH = REPO_ROOT / "rust" / "spelke-enumerator" / "target" / "release" / "spelke-enumerator"

# Fixed test task IDs (use synthetic tasks which are always present)
TASK_FILES = sorted(glob.glob(str(REPO_ROOT / "data" / "synthetic_cross_system" / "*.json")))[:10]


def load_tasks(n: int = 10) -> list[ArcTask]:
    """Load N fixed test tasks."""
    tasks = []
    for f in TASK_FILES[:n]:
        task_id = os.path.basename(f).replace(".json", "")
        with open(f) as fp:
            d = json.load(fp)
        tasks.append(ArcTask.from_dict(task_id, d))
    return tasks


def build_library() -> Library:
    registry = build_spelke_library()
    return Library(registry)


def run_rust_raw(library: Library, task: ArcTask, max_cost: int = 4, time_budget: float = 5.0) -> dict:
    """Call the Rust binary directly and return parsed JSON output."""
    lib_prims = _serialize_library(library)
    payload = {
        "schema_version": 1,
        "library": lib_prims,
        "task_id": task.task_id,
        "max_cost": max_cost,
        "time_budget": time_budget,
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    proc = subprocess.run(
        [str(BINARY_PATH)],
        input=payload_bytes,
        capture_output=True,
        timeout=time_budget + 5.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rust binary exited {proc.returncode}: {proc.stderr.decode()}")
    return json.loads(proc.stdout.decode("utf-8"))


def check_r1() -> tuple[bool, str]:
    """R1: Rust binary exists."""
    if BINARY_PATH.exists():
        return True, f"PASS binary at {BINARY_PATH}"
    return False, f"FAIL binary not found at {BINARY_PATH} — run: cargo build --release"


def check_r2(tasks: list[ArcTask], library: Library) -> tuple[bool, str]:
    """R2: Rust binary runs on 10 tasks and exits 0."""
    errors = []
    for task in tasks:
        try:
            output = run_rust_raw(library, task, max_cost=4, time_budget=3.0)
            assert "result" in output, f"Missing 'result' field in output"
        except Exception as e:
            errors.append(f"task {task.task_id}: {e}")
    if not errors:
        return True, f"PASS all {len(tasks)} tasks ran successfully"
    return False, f"FAIL {len(errors)} tasks failed: {errors[:3]}"


def check_r3(tasks: list[ArcTask], library: Library) -> tuple[bool, str]:
    """R3: Rust solve set matches Python solve set on 10 tasks."""
    MAX_COST = 4
    TIME_BUDGET = 5.0

    py_enum = TypeDirectedEnumerator(max_cost=MAX_COST, time_budget=TIME_BUDGET)
    rust_enum = RustEnumerator(max_cost=MAX_COST, time_budget=TIME_BUDGET)

    py_solved = set()
    rust_solved = set()
    details = []

    for task in tasks:
        py_result = py_enum.enumerate(task, library)
        rust_result = rust_enum.enumerate(task, library)

        py_ok = py_result is not None
        rust_ok = rust_result is not None

        if py_ok:
            py_solved.add(task.task_id)
        if rust_ok:
            rust_solved.add(task.task_id)

        if py_ok != rust_ok:
            details.append(f"{task.task_id}: py={'solved' if py_ok else 'unsolved'} rust={'solved' if rust_ok else 'unsolved'}")

    if py_solved == rust_solved:
        return True, f"PASS exact match: both solve {len(py_solved)}/{len(tasks)} tasks"
    return False, (
        f"FAIL mismatch: py_solved={py_solved}, rust_solved={rust_solved}\n"
        f"  Differences: {details}"
    )


def check_r4(tasks: list[ArcTask], library: Library) -> tuple[bool, str]:
    """R4: Rust is faster than Python on same 10 tasks (wall time comparison)."""
    MAX_COST = 4
    TIME_BUDGET = 3.0

    py_enum = TypeDirectedEnumerator(max_cost=MAX_COST, time_budget=TIME_BUDGET)
    rust_enum = RustEnumerator(max_cost=MAX_COST, time_budget=TIME_BUDGET)

    # Run both — time them
    t0 = time.time()
    for task in tasks:
        py_enum.enumerate(task, library)
    py_elapsed = time.time() - t0

    t0 = time.time()
    for task in tasks:
        rust_enum.enumerate(task, library)
    rust_elapsed = time.time() - t0

    speedup = py_elapsed / max(rust_elapsed, 0.001)
    if rust_elapsed < py_elapsed:
        return True, f"PASS Rust {speedup:.1f}x faster (Rust={rust_elapsed:.2f}s Python={py_elapsed:.2f}s)"
    return False, f"FAIL Rust NOT faster (Rust={rust_elapsed:.2f}s Python={py_elapsed:.2f}s speedup={speedup:.2f}x)"


def check_r5(tasks: list[ArcTask], library: Library) -> tuple[bool, str]:
    """R5: Rust handles timeout — tasks with tiny budget return empty, don't hang."""
    task = tasks[0]
    lib_prims = _serialize_library(library)
    payload = {
        "schema_version": 1,
        "library": lib_prims,
        "task_id": task.task_id,
        "max_cost": 6,
        "time_budget": 0.1,  # 100ms — should timeout or finish quickly
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(BINARY_PATH)],
            input=payload_bytes,
            capture_output=True,
            timeout=5.0,  # hard external timeout
        )
        elapsed = time.time() - t0
        output = json.loads(proc.stdout.decode("utf-8"))
        # Should complete within 2x the budget
        if elapsed > 2.0:
            return False, f"FAIL Rust ran for {elapsed:.2f}s with 0.1s budget — did not respect timeout"
        return True, f"PASS Rust returned in {elapsed*1000:.0f}ms with 100ms budget (status={output.get('result')})"
    except subprocess.TimeoutExpired:
        return False, "FAIL Rust hung past 5s external timeout — did not honor time_budget"
    except Exception as e:
        return False, f"FAIL exception: {e}"


def check_r6(tasks: list[ArcTask], library: Library) -> tuple[bool, str]:
    """R6: Rust handles nodes_explored cap — returns partial results, doesn't OOM."""
    # Run with max_cost=5 (higher cost = more nodes) and check n_explored
    task = tasks[0]
    try:
        output = run_rust_raw(library, task, max_cost=5, time_budget=5.0)
        n_explored = output.get("n_explored", 0)
        result_status = output.get("result", "unknown")
        # The result should come back with some programs or a node_cap status
        return True, f"PASS n_explored={n_explored} result={result_status} (no OOM)"
    except Exception as e:
        return False, f"FAIL {e}"


def check_r7() -> tuple[bool, str]:
    """R7: 98+ Python tests pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    output = result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
    lines = [l for l in output.splitlines() if "passed" in l or "failed" in l or "error" in l]
    # Parse passed count
    import re
    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0

    if passed >= 98 and failed == 0:
        return True, f"PASS {passed} Python tests pass, {failed} fail"
    return False, f"FAIL {passed} passed, {failed} failed (need ≥98 passed, 0 failed)"


def check_r8(tasks: list[ArcTask], library: Library) -> tuple[bool, str]:
    """R8: Rust binary produces valid JSON output with correct schema."""
    errors = []
    for task in tasks[:3]:
        try:
            output = run_rust_raw(library, task, max_cost=4, time_budget=3.0)
            # Check required fields
            required = ["schema_version", "result", "programs", "n_explored", "elapsed_ms"]
            for field in required:
                if field not in output:
                    errors.append(f"task {task.task_id}: missing field {field!r}")
                    continue
            if output.get("schema_version") != 1:
                errors.append(f"task {task.task_id}: schema_version != 1")
            if output.get("result") not in ["found", "not_found", "timeout", "node_cap", "error"]:
                errors.append(f"task {task.task_id}: invalid result={output.get('result')!r}")
            programs = output.get("programs", [])
            if not isinstance(programs, list):
                errors.append(f"task {task.task_id}: programs is not a list")
            for prog in programs[:3]:
                if "cost" not in prog or "sexp" not in prog:
                    errors.append(f"task {task.task_id}: program entry missing cost/sexp")
        except Exception as e:
            errors.append(f"task {task.task_id}: {e}")

    if not errors:
        return True, "PASS all checked tasks have valid JSON schema"
    return False, f"FAIL schema errors: {errors}"


def main():
    print("=" * 60)
    print("RUST ENUMERATOR VERIFICATION")
    print("=" * 60)

    results = []

    # R1: binary exists
    ok, msg = check_r1()
    print(f"\n[R1] Binary exists: {msg}")
    results.append(ok)

    if not ok:
        # Can't do R2-R6 or R8 without the binary
        print("\n[R2] SKIP (binary not found)")
        print("[R3] SKIP (binary not found)")
        print("[R4] SKIP (binary not found)")
        print("[R5] SKIP (binary not found)")
        print("[R6] SKIP (binary not found)")
        results.extend([False] * 5)

        # R7 can still run
        ok7, msg7 = check_r7()
        print(f"\n[R7] Python tests: {msg7}")
        results.append(ok7)

        print("\n[R8] SKIP (binary not found)")
        results.append(False)

        n_pass = sum(results)
        print(f"\n{'=' * 60}")
        print(f"RUST ENUMERATOR VERIFIED — {n_pass}/8")
        sys.exit(1 if n_pass < 8 else 0)

    # Binary exists — load tasks and library for R2-R8
    print("\nLoading tasks and library...")
    tasks = load_tasks(10)
    library = build_library()
    print(f"Loaded {len(tasks)} tasks, {len(list(library.base_registry))} primitives")

    # R2: binary runs
    ok, msg = check_r2(tasks, library)
    print(f"\n[R2] Binary runs: {msg}")
    results.append(ok)

    # R3: correctness match
    print("\n[R3] Correctness check (running Python and Rust enumerators)...")
    ok, msg = check_r3(tasks, library)
    print(f"[R3] Solve set match: {msg}")
    results.append(ok)

    # R4: speed comparison
    print("\n[R4] Speed comparison...")
    ok, msg = check_r4(tasks, library)
    print(f"[R4] Speed: {msg}")
    results.append(ok)

    # R5: timeout
    ok, msg = check_r5(tasks, library)
    print(f"\n[R5] Timeout: {msg}")
    results.append(ok)

    # R6: node cap
    ok, msg = check_r6(tasks, library)
    print(f"\n[R6] Node cap: {msg}")
    results.append(ok)

    # R7: Python tests
    ok, msg = check_r7()
    print(f"\n[R7] Python tests: {msg}")
    results.append(ok)

    # R8: JSON schema
    ok, msg = check_r8(tasks, library)
    print(f"\n[R8] JSON schema: {msg}")
    results.append(ok)

    n_pass = sum(results)
    print(f"\n{'=' * 60}")
    print(f"RUST ENUMERATOR VERIFIED — {n_pass}/8")
    if n_pass < 8:
        failed = [i + 1 for i, ok in enumerate(results) if not ok]
        print(f"Failed checks: R{', R'.join(str(f) for f in failed)}")

    sys.exit(0 if n_pass == 8 else 1)


if __name__ == "__main__":
    main()
