#!/usr/bin/env python3
"""
verify_arxiv_ready.py — 10-check gate before arXiv submission.

Usage:
    cd bootstrap-substrate
    python3 experiments/scripts/verify_arxiv_ready.py

Exits 1 if any check fails.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PAPER_HTML = ROOT / "paper" / "paper.html"
PAPER_PDF = ROOT / "paper" / "paper.pdf"
STAT_JSON = ROOT / "experiments/outputs/stat_significance_5seed/statistical_significance.json"

results = []

def check(label, passed, reason=""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {label}"
    if reason:
        msg += f" — {reason}"
    print(msg)
    results.append(passed)

html = PAPER_HTML.read_text(encoding="utf-8") if PAPER_HTML.exists() else ""

# 1a: paper.html contains "p=0.2482" AND "McNemar" AND ("χ²" or "chi2")
has_p = "p=0.2482" in html
has_mcnemar = "McNemar" in html
has_chi = "χ²" in html or "chi2" in html
check("1a: p=0.2482 + McNemar + χ² present in paper.html",
      has_p and has_mcnemar and has_chi,
      f"p=0.2482={has_p}, McNemar={has_mcnemar}, χ²/chi2={has_chi}")

# 1b: paper.html contains sample efficiency task counts — 10, 25, 50 all present
# They must appear in the context of the sample efficiency section
def near(text, keyword, window=2000):
    idx = text.find(keyword)
    while idx != -1:
        snippet = text[max(0,idx-window):idx+window]
        yield snippet
        idx = text.find(keyword, idx+1)

se_text = "\n".join(near(html, "ample efficiency", 3000))
has_10 = ">10<" in se_text or "N=10" in se_text or ">10</td>" in se_text or "n=10" in se_text.lower() or "10</td>" in se_text
has_25 = ">25<" in se_text or "N=25" in se_text or ">25</td>" in se_text or "n=25" in se_text.lower() or "25</td>" in se_text
has_50 = ">50<" in se_text or "N=50" in se_text or ">50</td>" in se_text or "n=50" in se_text.lower() or "50</td>" in se_text
check("1b: Sample efficiency task counts 10/25/50 present",
      has_10 and has_25 and has_50,
      f"10={has_10}, 25={has_25}, 50={has_50}")

# 1c: paper.html contains AGENTS+PLACES result
has_agents_places = ("AGENTS" in html and "PLACES" in html) and ("45/400" in html or "11.2%" in html or "11.25%" in html or "45 /" in html)
check("1c: AGENTS+PLACES result present (honest)",
      has_agents_places,
      "need AGENTS, PLACES, and 45/400 or 11.2%")

# 1d: paper.pdf mtime >= paper.html mtime
if PAPER_PDF.exists() and PAPER_HTML.exists():
    pdf_newer = PAPER_PDF.stat().st_mtime >= PAPER_HTML.stat().st_mtime
    check("1d: paper.pdf mtime >= paper.html mtime", pdf_newer,
          f"pdf={PAPER_PDF.stat().st_mtime:.0f} html={PAPER_HTML.stat().st_mtime:.0f}")
else:
    check("1d: paper.pdf mtime >= paper.html mtime", False, "pdf or html missing")

# 1e: abstract contains "not statistically significant" OR "p=0.2482"
abstract_start = html.find('class="paper-summary-block"')
abstract_end = html.find('class="two-col"')
abstract_text = html[abstract_start:abstract_end] if abstract_start != -1 and abstract_end != -1 else ""
has_not_sig = "not statistically significant" in abstract_text or "p=0.2482" in abstract_text
check("1e: Abstract contains 'not statistically significant' or 'p=0.2482'",
      has_not_sig, f"abstract snippet: '{abstract_text[200:300].strip()}'")

# 1f: within 500 chars of "Limitation" — contains "not statistically significant" or "p=0.2482"
lim_idx = html.find("Limitation")
if lim_idx != -1:
    lim_window = html[lim_idx:lim_idx+2000]
    has_lim_stat = "not statistically significant" in lim_window or "p=0.2482" in lim_window
    check("1f: Limitations section contains not-significant / p=0.2482", has_lim_stat)
else:
    check("1f: Limitations section contains not-significant / p=0.2482", False, "'Limitation' not found")

# 1g: paper.html contains compounding depth result — "b9b7f026" or "4→2" or "depth 2"
has_compound = "b9b7f026" in html and ("4→2" in html or "depth 2" in html or "depth=2" in html or "depth 4" in html)
check("1g: Compounding depth result for b9b7f026 present",
      has_compound, f"b9b7f026={'b9b7f026' in html}, depth marker={'4→2' in html or 'depth 2' in html}")

# 1h: statistical_significance.json has n_seeds == 5
if STAT_JSON.exists():
    stat = json.loads(STAT_JSON.read_text())
    svg = stat.get("spelke_vs_generic", {})
    n_seeds = svg.get("n_seeds", stat.get("n_seeds", None))
    check("1h: statistical_significance.json n_seeds == 5", n_seeds == 5, f"n_seeds={n_seeds}")
else:
    check("1h: statistical_significance.json n_seeds == 5", False, "file not found")

# 1i: pytest 98+ pass, 0 failures
try:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    out = r.stdout + r.stderr
    import re
    m = re.search(r"(\d+) passed", out)
    n_pass = int(m.group(1)) if m else 0
    fail_m = re.search(r"(\d+) failed", out)
    n_fail = int(fail_m.group(1)) if fail_m else 0
    check("1i: pytest 98+ pass, 0 failures", n_pass >= 98 and n_fail == 0,
          f"{n_pass} passed, {n_fail} failed")
except Exception as e:
    check("1i: pytest 98+ pass, 0 failures", False, str(e))

# 1j: zero editorial artifacts (TODO/FIXME/TBD/INSERT/once the experiment/id="*-placeholder")
import re as _re
editorial_patterns = [
    r'\bTODO\b', r'\bFIXME\b', r'\bTBD\b',
    r'INSERT DATA', r'once the experiment completes',
    r'data loading from experiments',
    r'id="[^"]*-placeholder"',
    r'<!--.*(?:todo|fixme|insert)',
]
editorial_re = _re.compile('|'.join(editorial_patterns), _re.IGNORECASE)
matches = editorial_re.findall(html)
check("1j: Zero editorial artifacts (TODO/FIXME/placeholder-id/etc) in paper.html",
      len(matches) == 0, f"found {len(matches)} matches: {matches[:3]}")

# Summary
n_pass = sum(results)
n_total = len(results)
print(f"\n{'='*50}")
if n_pass == n_total:
    print(f"ARXIV READY — {n_pass}/{n_total} passed")
else:
    print(f"NOT READY — {n_pass}/{n_total} passed, {n_total - n_pass} failed")
sys.exit(0 if n_pass == n_total else 1)
