"""
verify_phase1.py — Phase 1 completion checks for dated_implementation.md.

Run from bootstrap-substrate/:
    python3 experiments/scripts/verify_phase1.py

All 5 steps must pass for Phase 1 to be considered complete.
"""
import json
import glob
import os
import time
import sys

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        PASS.append(name)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        FAIL.append(name)


# ── STEP 1: Statistical significance ──────────────────────────────────────────

def check_1a():
    d = json.load(open("experiments/outputs/stat_significance_5seed/checkpoint.json"))
    assert d["done"] == 15, f"Only {d['done']}/15 done"
    r = {(x["seed"], x["dsl"]): x["n_solved"] for x in d["results_so_far"]}
    for s in range(5):
        sp = r[(s, "spelke")]
        ge = r[(s, "generic")]
        vi = r[(s, "vimrl")]
        print(f"      seed={s}: spelke={sp} generic={ge} vimrl={vi}")
        assert sp >= 40, f"seed={s} spelke suspiciously low: {sp}"
        assert ge >= 38, f"seed={s} generic suspiciously low: {ge}"

def check_1b():
    d = json.load(open("experiments/outputs/stat_significance_5seed/statistical_significance.json"))
    sg = d["spelke_vs_generic"]
    sv = d["spelke_vs_vimrl"]
    print(f"      Spelke vs Generic: chi2={sg['chi2']:.4f}, p={sg['p_value']:.4f}")
    print(f"      Spelke vs VIMRL:   chi2={sv['chi2']:.4f}, p={sv['p_value']:.4f}")
    assert isinstance(sg["chi2"], float) and sg["chi2"] > 0
    assert 0 < sg["p_value"] < 1
    assert sg.get("n_seeds", 0) == 5, "must be from 5 seeds"


# ── STEP 2: Sample efficiency ─────────────────────────────────────────────────

def check_2a():
    files = glob.glob("experiments/outputs/sample_efficiency*/sample_efficiency_results.json")
    assert files, "No sample_efficiency_results.json found"
    d = json.load(open(sorted(files)[-1]))
    assert d["config"]["seeds"] == 3
    assert d["config"]["task_counts"] == [10, 25, 50, 100, 200, 400]
    spelke_rates = [e["mean_solve_rate"] for e in d["spelke"]]
    generic_rates = [e["mean_solve_rate"] for e in d["generic"]]
    print(f"      Spelke rates: {[f'{r:.1%}' for r in spelke_rates]}")
    print(f"      Generic rates: {[f'{r:.1%}' for r in generic_rates]}")
    assert spelke_rates[0] != spelke_rates[-1] or spelke_rates[0] == 0.0, \
        "Spelke rate identical at n=10 and n=400 — suspicious"
    assert len(spelke_rates) == 6

def check_2b():
    import subprocess
    r = subprocess.run(
        ["grep", "-c", r"sample.effic\|task_count\|10.*25.*50\|Figure.*efficiency", "paper/paper.html"],
        capture_output=True, text=True
    )
    count = int(r.stdout.strip() or "0")
    assert count > 0, "No sample efficiency content found in paper.html"


# ── STEP 3: AGENTS+PLACES ─────────────────────────────────────────────────────

def check_3a():
    files = glob.glob("experiments/outputs/phase2_spelke/**/*.json", recursive=True)
    assert files, "No phase2_spelke output files"
    r = [f for f in files if "experiment_results" in f]
    assert r, "No experiment_results.json in phase2_spelke"
    d = json.load(open(r[0]))
    rate = d.get("solve_rate") or d.get("final_solve_rate")
    size = d.get("final_library_size")
    print(f"      Phase2 solve rate: {rate}, library: {size}")

def check_3b():
    import subprocess
    r = subprocess.run(
        ["grep", "-c", r"AGENTS\|PLACES\|full.spelke\|170.prim\|extended.DSL\|Extended DSL", "paper/paper.html"],
        capture_output=True, text=True
    )
    count = int(r.stdout.strip() or "0")
    assert count > 0, "AGENTS+PLACES not mentioned in paper.html"


# ── STEP 4: Paper revision ────────────────────────────────────────────────────

def check_4a():
    d = json.load(open("experiments/outputs/stat_significance_5seed/statistical_significance.json"))
    p = d["spelke_vs_generic"]["p_value"]
    html = open("paper/paper.html").read()
    p_str = f"{p:.4f}"
    assert p_str in html or f"{p:.3f}" in html, f"p-value {p_str} not in paper.html"
    assert "McNemar" in html

def check_4b():
    html = open("paper/paper.html").read()
    idx = html.lower().find("abstract")
    chunk = html[idx:idx+2000].lower()
    assert any(w in chunk for w in ["statistic", "p-value", "significant", "mcnemar"]), \
        "Abstract not updated with stats"

def check_4c():
    pdf_mt = os.path.getmtime("paper/paper.pdf")
    html_mt = os.path.getmtime("paper/paper.html")
    age_h = (time.time() - pdf_mt) / 3600
    print(f"      PDF age: {age_h:.1f}h")
    assert pdf_mt >= html_mt, "PDF older than HTML"
    assert age_h < 48, "PDF not regenerated in this session"


# ── STEP 5: Integrity ─────────────────────────────────────────────────────────

def check_5a():
    import subprocess
    r = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True
    )
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print(f"      {last}")
    assert "failed" not in last and "error" not in last.lower(), f"Tests failed: {last}"
    assert "passed" in last


# ── Main ──────────────────────────────────────────────────────────────────────

print("\nPhase 1 Verification — dated_implementation.md\n")
print("STEP 1: Statistical Significance")
check("1a: 15 runs complete", check_1a)
check("1b: McNemar p-value real", check_1b)

print("\nSTEP 2: Sample Efficiency")
check("2a: sample_efficiency_results.json valid", check_2a)
check("2b: paper contains efficiency data", check_2b)

print("\nSTEP 3: AGENTS+PLACES")
check("3a: phase2_spelke experiment done", check_3a)
check("3b: result in paper", check_3b)

print("\nSTEP 4: Paper Revision")
check("4a: p-value in paper verbatim", check_4a)
check("4b: abstract updated", check_4b)
check("4c: PDF regenerated", check_4c)

print("\nSTEP 5: Integrity")
check("5a: all tests pass", check_5a)

print(f"\n{'='*50}")
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print(f"  FAILED: {', '.join(FAIL)}")
    sys.exit(1)
else:
    print("  PHASE 1 COMPLETE")
