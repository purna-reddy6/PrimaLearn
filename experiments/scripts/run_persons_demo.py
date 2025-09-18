"""
run_persons_demo.py — PERSONS curriculum demo for P6 verification.

Loads 30 synthetic_persons_objects tasks, runs AST solver with PERSONS
library, compresses with Stitch, and verifies ≥1 is_cross_system abstraction
with PERSONS in systems_composed.

Output: experiments/outputs/persons_curriculum_demo/curriculum_demo_results.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.spelke_dsl import build_spelke_library
from src.engine.library import Library
from src.engine.ast_solver import ASTSolver
from src.engine.stitch import StitchCompressor
from src.arc.grid import ArcTask

ROOT = Path(__file__).parent.parent.parent
PERSONS_DIR = ROOT / "data" / "synthetic_persons_objects"
OUT_DIR = ROOT / "experiments" / "outputs" / "persons_curriculum_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tasks(directory):
    tasks = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        tid = fname.replace(".json", "")
        with open(os.path.join(directory, fname)) as f:
            data = json.load(f)
        tasks.append(ArcTask.from_dict(tid, data))
    return tasks


def classify_program(prog, library):
    """Return the set of Spelke systems used by a program."""
    systems = set()
    for pname in prog.root.primitives_used():
        if pname in library.base_registry._primitives:
            p = library.base_registry[pname]
            systems.add(p.system.name)
    return systems


def run_persons_demo():
    print("=" * 70)
    print("  PERSONS CURRICULUM DEMO — Cross-System Abstraction Check")
    print("=" * 70)

    # Load tasks
    tasks = load_tasks(PERSONS_DIR)
    print(f"\nLoaded {len(tasks)} PERSONS+OBJECTS tasks from {PERSONS_DIR}")

    # Build library with PERSONS
    reg = build_spelke_library(include_persons=True)
    lib = Library(registry=reg)
    persons_prims = [p for p in reg if p.system.name == 'PERSONS']
    print(f"Library: {len(list(reg._primitives))} primitives (PERSONS: {len(persons_prims)})")

    # Solve tasks
    solver = ASTSolver(lib)
    programs = []
    solved_ids = []
    print("\nSolving tasks...")
    for task in tasks:
        prog = solver.solve(task)
        if prog:
            programs.append(prog)
            solved_ids.append(task.task_id)

    print(f"Solved: {len(solved_ids)}/{len(tasks)}")

    # Classify by system
    system_counts = defaultdict(int)
    cross_system_programs = []
    for prog in programs:
        sys_used = classify_program(prog, lib)
        key = "+".join(sorted(sys_used))
        system_counts[key] += 1
        if len(sys_used) > 1:
            cross_system_programs.append(prog)
    print(f"\nProgram distribution:")
    for key, cnt in sorted(system_counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {cnt}")
    print(f"Cross-system programs (≥2 systems): {len(cross_system_programs)}")

    # Run Stitch compression
    print("\nRunning Stitch compression...")
    compressor = StitchCompressor(lib, max_abstractions=15)
    abstractions = compressor.compress(programs)

    # Check for PERSONS cross-system abstractions
    persons_cross_abs = []
    for a in abstractions:
        if a.is_cross_system and any(s.upper() == "PERSONS" for s in a.systems_composed):
            persons_cross_abs.append(a)

    print(f"\nAbstractions found: {len(abstractions)}")
    print(f"Cross-system abstractions: {len([a for a in abstractions if a.is_cross_system])}")
    print(f"Cross-system with PERSONS: {len(persons_cross_abs)}")

    for a in abstractions:
        body_str = a.body.to_str() if a.body else "?"
        marker = " ← PERSONS CROSS-SYSTEM" if a in persons_cross_abs else ""
        print(f"  {a.name}: systems={sorted(a.systems_composed)}, "
              f"cross={a.is_cross_system}, savings={a.mdl_savings:.0f}{marker}")
        print(f"    body: {body_str[:100]}")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks_total": len(tasks),
        "tasks_solved": len(solved_ids),
        "spelke": {
            "cycle_0": {
                "synthetic_solved": len(solved_ids),
                "abstractions": len(abstractions),
                "cross_system_abstractions": len([a for a in abstractions if a.is_cross_system]),
                "persons_cross_system_abstractions": len(persons_cross_abs),
                "abstraction_details": [
                    {
                        "name": a.name,
                        "body": a.body.to_str() if a.body else None,
                        "systems": sorted(a.systems_composed),
                        "is_cross_system": a.is_cross_system,
                        "savings": a.mdl_savings,
                    }
                    for a in abstractions
                ],
            }
        }
    }

    out_path = OUT_DIR / "curriculum_demo_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if persons_cross_abs:
        print("\n✓ P6 PASS: Found cross-system abstraction(s) with PERSONS in systems_composed")
        for a in persons_cross_abs:
            print(f"  {a.name}: systems={sorted(a.systems_composed)}")
    else:
        print("\n✗ P6 FAIL: No cross-system abstraction with PERSONS found")
        if abstractions:
            all_systems = {s for a in abstractions for s in a.systems_composed}
            print(f"  Systems in all abstractions: {sorted(all_systems)}")
            print("  Check: do TYPE F programs include PERSONS+NUMBER+OBJECTS primitives?")
            for prog in cross_system_programs[:3]:
                print(f"  Program: {prog.root.to_str() if hasattr(prog.root, 'to_str') else str(prog)}")

    return results


if __name__ == "__main__":
    run_persons_demo()
