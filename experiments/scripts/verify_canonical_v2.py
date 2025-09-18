"""
verify_canonical_v2.py — Check all C1-C8 completion conditions from AGENT_MISSION.md
against canonical_run_v2 outputs.

Usage:
    python3 experiments/scripts/verify_canonical_v2.py
"""
from __future__ import annotations
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CANONICAL_V2 = ROOT / "experiments/outputs/canonical_run_v2"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def check_all() -> dict[str, str]:
    results = {}

    # C1: canonical_run_v2/spelke/experiment_results.json exists
    spelke_results = load_json(CANONICAL_V2 / "spelke/experiment_results.json")
    if spelke_results is None:
        results["C1"] = "FAIL: canonical_run_v2/spelke/experiment_results.json missing"
        return results  # early exit — downstream checks need this
    else:
        results["C1"] = f"PASS: file exists, final_n_solved={spelke_results.get('final_n_solved')}"

    # C2: LAST cycle n_solved STRICTLY GREATER than FIRST cycle n_solved
    cycles = spelke_results.get("cycles", [])
    if len(cycles) < 2:
        results["C2"] = f"FAIL: need ≥2 cycles, got {len(cycles)}"
    else:
        first_n = cycles[0]["n_solved"]
        last_n = cycles[-1]["n_solved"]
        if last_n > first_n:
            results["C2"] = f"PASS: last cycle n_solved={last_n} > first cycle n_solved={first_n} (compounding confirmed)"
        else:
            results["C2"] = f"FAIL: last={last_n} NOT > first={first_n} — compounding not demonstrated"

    # C3: Spelke final_n_solved > Generic final_n_solved
    generic_results = load_json(CANONICAL_V2 / "generic/experiment_results.json")
    if spelke_results is None or generic_results is None:
        results["C3"] = f"FAIL: missing data"
    else:
        s = spelke_results.get("final_n_solved", 0)
        g = generic_results.get("final_n_solved", 0)
        if s > g:
            results["C3"] = f"PASS: Spelke={s} > Generic={g}"
        else:
            results["C3"] = f"FAIL: Spelke={s} <= Generic={g}"

    # C4: canonical_run_v2/generic/ and vimrl/ experiment_results.json exist
    vimrl_results = load_json(CANONICAL_V2 / "vimrl/experiment_results.json")
    parts = []
    if generic_results is not None:
        parts.append("generic ✓")
    else:
        parts.append("generic MISSING")
    if vimrl_results is not None:
        parts.append("vimrl ✓")
    else:
        parts.append("vimrl MISSING")
    if "MISSING" not in " ".join(parts):
        results["C4"] = f"PASS: {', '.join(parts)}"
    else:
        results["C4"] = f"FAIL: {', '.join(parts)}"

    # C5: paper/paper.html contains 'canonical_run_v2'
    paper_path = ROOT / "paper/paper.html"
    paper_text = paper_path.read_text() if paper_path.exists() else ""
    if "canonical_run_v2" in paper_text:
        count = paper_text.count("canonical_run_v2")
        results["C5"] = f"PASS: paper.html contains canonical_run_v2 ({count} occurrences)"
    else:
        results["C5"] = "FAIL: paper.html does not contain canonical_run_v2"

    # C6: results_summary.md contains 'canonical_run_v2'
    rs_path = ROOT / "results_summary.md"
    rs_text = rs_path.read_text() if rs_path.exists() else ""
    if "canonical_run_v2" in rs_text:
        results["C6"] = "PASS: results_summary.md contains canonical_run_v2"
    else:
        results["C6"] = "FAIL: results_summary.md does not contain canonical_run_v2"

    # C7: git log >= 3 commits
    result = subprocess.run(
        ["git", "log", "--oneline", "-20"],
        capture_output=True, text=True, cwd=ROOT
    )
    commits = [l for l in result.stdout.strip().split("\n") if l]
    if len(commits) >= 3:
        results["C7"] = f"PASS: {len(commits)} commits visible"
    else:
        results["C7"] = f"FAIL: only {len(commits)} commits"

    # C8: pytest passes
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=ROOT
    )
    if result.returncode == 0:
        results["C8"] = "PASS: pytest exits 0"
    else:
        results["C8"] = f"FAIL: pytest exit={result.returncode}\n{result.stdout[-500:]}"

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CANONICAL_RUN_V2 COMPLETION VERIFICATION (C1-C8)")
    print("=" * 60)
    results = check_all()
    all_pass = True
    for key in sorted(results.keys()):
        val = results[key]
        status = "✓" if val.startswith("PASS") else "✗"
        print(f"  {status} {key}: {val}")
        if not val.startswith("PASS"):
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("ALL C1-C8 CONDITIONS MET — DONE!")
    else:
        n_fail = sum(1 for v in results.values() if not v.startswith("PASS"))
        print(f"{n_fail} condition(s) not yet met.")
    sys.exit(0 if all_pass else 1)
