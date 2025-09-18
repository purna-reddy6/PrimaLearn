# PrimaLearn in Neural Program Synthesis

**Spelke-Initialized Library Learning Discovers Cross-Domain Abstractions**

*Purna Sainath Reddy V — Manipal Institute of Technology Bengaluru*

> Paper: [`paper/paper.pdf`](paper/paper.pdf)

---

## Overview

This repository contains the full implementation for the paper *"PrimaLearn in Neural Program Synthesis: Spelke-Initialized Library Learning Discovers Cross-Domain Abstractions"*.

We present **bootstrap-substrate**, the first end-to-end computational instantiation of Carey's PrimaLearn theory within a neural program-synthesis architecture. The system initializes a typed domain-specific language (DSL) with primitives from Spelke's core-knowledge systems (OBJECTS, FORMS, NUMBER) and runs a DreamCoder-style wake-sleep loop with Stitch anti-unification over the ARC-AGI benchmark.

### Three main results

1. **Cross-system abstraction discovery.** The MDL compressor discovers abstractions bridging FORMS and OBJECTS — on synthetic tasks (`abs₀,₀`, MDL savings=124) and on real ARC tasks (`abs₀,₂`, MDL savings=25, reused across 4 programs).

2. **Compounding.** On the Spelke Bootstrap Suite v2 (875 tasks, §4.3), 75 tasks structurally unsolvable in cycle 0 become solvable in cycle 1 via cross-system abstractions discovered in cycle 0. Solve rate rises 91.4% → 100.0%. On ARC, task `b9b7f026` is first solved in cycle 1 via composition with a learned OBJECTS abstraction.

3. **Three-way baseline comparison.** Spelke (143 prims) achieves 46/400 (11.5%) on ARC-AGI vs. 42/400 (10.5%) for a matched-cardinality Generic DSL and 42/400 (10.5%) for the VIMRL objectness-only DSL. The library grows from 143 → 171 primitives across 3 cycles; baseline libraries remain static.

---

## Setup

```bash
# Clone and install
git clone https://github.com/purna-reddy6/PrimaLearn.git
cd PrimaLearn
pip install -e ".[dev]"

# (Optional) Build Rust enumerator for 4.6–9.4× speedup
cd rust/spelke-enumerator && cargo build --release
```

**ARC-AGI data** is included in `data/arc-agi-1/` (Apache 2.0 license, François Chollet).

---

## Reproducing the paper results

### Canonical ARC result (§4.4, Tables 4–6)

```bash
python experiments/scripts/run_experiment.py \
  --cycles 3 --enum-cost 5 --enum-budget 20 \
  --run-generic --run-vimrl \
  --output-dir experiments/outputs/canonical_run_v2 --verbose
```

### Compounding benchmark v2 (§4.3, Table 3)

```bash
python experiments/scripts/run_experiment.py \
  --task-dir data/spelke_benchmark_v2/train \
  --cycles 5 --enum-cost 6 --enum-budget 15 --rust --full-spelke \
  --output-dir experiments/outputs/benchmark_v2_run
```

### Cross-system demo (§4.2, Table 2)

```bash
python experiments/scripts/run_cross_system_demo.py
```

### Curriculum pre-training (§4.8)

```bash
python experiments/scripts/run_experiment.py \
  --task-dir data/synthetic_forms_objects_places \
              data/synthetic_number_objects \
              data/synthetic_number_objects_replicate \
              data/synthetic_count_cells \
  --cycles 1 --enum-cost 6 --enum-budget 10 --rust \
  --output-dir experiments/outputs/curriculum_v2_arc_run/pretrain
```

### PERSONS system (§4.9)

```bash
python experiments/scripts/run_persons_demo.py
```

### Statistical significance (§4.4)

```bash
python experiments/scripts/run_statistical_significance.py
```

### Sample efficiency curves (§4.6, Table 7, Figure 5)

```bash
python experiments/scripts/sample_efficiency.py
```

### Verify all canonical conditions

```bash
python experiments/scripts/verify_canonical_v2.py  # 8 conditions
python experiments/scripts/verify_benchmark_v2.py  # 10/10
python experiments/scripts/verify_phase1.py        # 10/10
python experiments/scripts/verify_arxiv_ready.py   # 10/10
python experiments/scripts/verify_curriculum.py    # 12/12
python experiments/scripts/verify_transfer.py      # 5/5
python experiments/scripts/verify_persons.py       # 10/10
```

### Run all tests

```bash
pytest tests/ -v  # 220/220
```

---

## Repository structure

```
bootstrap-substrate/
├── paper/                        # Paper PDF and figures
│   ├── paper.pdf
│   └── fig{1-6}_*.png
├── src/
│   ├── spelke_dsl/               # The Spelke-typed DSL (143 primitives)
│   │   ├── l_objects.py          # OBJECTS: 38 prims (cohesion, gravity, flood_fill, ...)
│   │   ├── l_forms.py            # FORMS: 31 prims (rotate, flip, tile, symmetry, ...)
│   │   ├── l_number.py           # NUMBER: 41 prims (OTS cap-3, ANS Weber ratio, ...)
│   │   ├── l_agents.py           # AGENTS: 11 prims (pathfinding, goal inference, ...)
│   │   ├── l_places.py           # PLACES: 16 prims (quadrants, distance fields, ...)
│   │   ├── l_persons.py          # PERSONS: 12 prims (intentionality, social distance, ...)
│   │   ├── glue.py               # GLUE: 33 higher-order combinators + NSM operators
│   │   └── analogy.py            # Gentner analogy alignment primitive
│   ├── engine/
│   │   ├── ast_solver.py         # ASTSolver: 22+ strategy families
│   │   ├── enumerator.py         # TypeDirectedEnumerator: cost-bounded bottom-up search
│   │   ├── stitch.py             # StitchCompressor + MDL accept + cross-cycle dedup
│   │   ├── wake_sleep.py         # WakeSleepEngine: full wake-sleep loop
│   │   ├── dream_generator.py    # DreamGenerator: 30% pipeline + 30% cross-sys + 40% PCFG
│   │   ├── recognition.py        # NeuralRecognitionNet: 2-layer NumPy MLP
│   │   ├── library.py            # Library: growing combinator store with MDL tracking
│   │   ├── program.py            # ProgramNode AST: PrimNode, AppNode, LamNode, ...
│   │   ├── language.py           # LAPS-style language conditioning
│   │   └── rust_enumerator.py    # Python bridge to Rust enumerator
│   ├── arc/
│   │   ├── evaluator.py          # TaskResult, EvalResults, Evaluator
│   │   ├── grid.py               # Grid primitives
│   │   └── loader.py             # ARC task loader
│   ├── baselines/
│   │   ├── generic_dsl.py        # Generic DSL (143 prims, matched cardinality)
│   │   └── vimrl_dsl.py          # VIMRL objectness-only DSL (Ainooson 2023, 45 prims)
│   └── analysis/
│       └── carey_signature.py    # Carey signature detection (systems_composed tracking)
├── rust/
│   └── spelke-enumerator/        # Rust port: 4.6× (cost=5) to 9.4× (cost=6) speedup
├── data/
│   ├── arc-agi-1/                # ARC-AGI training set (400 tasks)
│   ├── synthetic_cross_system/   # 15 synthetic FORMS+OBJECTS tasks (§4.2)
│   ├── synthetic_compounding/    # 5 compounding tasks (§4.1)
│   ├── spelke_benchmark_v2/      # 875 train + 200 test — Spelke Bootstrap Suite v2 (§4.3)
│   ├── synthetic_forms_objects_places/   # 30 curriculum tasks (§4.8)
│   ├── synthetic_number_objects/         # 30 curriculum tasks (§4.8)
│   ├── synthetic_number_objects_replicate/ # 30 curriculum tasks (§4.8)
│   ├── synthetic_count_cells/            # 30 curriculum tasks (§4.8)
│   └── synthetic_persons_objects/        # 30 PERSONS tasks (§4.9)
├── experiments/
│   ├── scripts/                  # All experiment and verification scripts
│   └── outputs/                  # Canonical result data for all paper tables/figures
│       ├── canonical_run_v2/     # §4.4–4.6: 3-way ARC comparison (46/42/42)
│       ├── benchmark_v2_run/     # §4.3: compounding 91.4%→100.0%
│       ├── cross_system_demo/    # §4.2: abs₀,₀ FORMS+OBJECTS (MDL=124)
│       ├── sample_efficiency_full_20260526_205111/  # §4.6: Table 7 data
│       ├── stat_significance_5seed/  # §4.4: McNemar χ²=1.3333, p=0.2482
│       ├── phase2_spelke/        # §4.7: AGENTS+PLACES null result (45/400)
│       ├── curriculum_v2_arc_run/ # §4.8: curriculum transfer (47/400)
│       └── persons_curriculum_demo/ # §4.9: PERSONS 30/30 vs Generic 0/30
└── tests/                        # 220 tests
```

---

## Key results

| Experiment | Result | Data |
|-----------|--------|------|
| ARC-AGI 3-way comparison | Spelke 46/400 (11.5%) vs Generic 42/400 vs VIMRL 42/400 | `outputs/canonical_run_v2/` |
| Cross-system abstraction (synthetic) | abs₀,₀ FORMS+OBJECTS, MDL savings=124 | `outputs/cross_system_demo/` |
| Cross-system abstraction (ARC) | abs₀,₂ FORMS+OBJECTS, MDL savings=25, reuse=4 | `outputs/canonical_run_v2/` |
| Compounding benchmark v2 | 800/875 cycle 0 → 875/875 cycle 1 (75 Tier 3 unlocked) | `outputs/benchmark_v2_run/` |
| Compounding on ARC | b9b7f026 unsolvable cycle 0, solved cycle 1 via abs₀,₁₀ | `outputs/canonical_run_v2/` |
| Statistical significance | McNemar χ²=1.3333, p=0.2482 (5 seeds, N=400) | `outputs/stat_significance_5seed/` |
| Sample efficiency | Spelke advantage emerges at N≥50, peaks +1.7% at N=100 | `outputs/sample_efficiency_full_20260526_205111/` |
| AGENTS+PLACES (170 prims) | 45/400 (null result, ARC has no agent/nav tasks) | `outputs/phase2_spelke/` |
| Curriculum transfer | 47/400 (+1 via d631b094, NUMBER+OBJECTS cross-sys) | `outputs/curriculum_v2_arc_run/` |
| PERSONS (6th system) | 30/30 vs Generic 0/30 on PERSONS curriculum | `outputs/persons_curriculum_demo/` |

---

## Citation

```bibtex
@article{corry2026quinian,
  title   = {PrimaLearn in Neural Program Synthesis:
             Spelke-Initialized Library Learning Discovers Cross-Domain Abstractions},
  author  = {Reddy V, Purna Sainath},
  year    = {2026},
  note    = {Preprint}
}
```

---

## References

1. Ainooson et al. (2023). VIMRL: Objectness-centric DSL for ARC.
2. Carey (2009). *The Origin of Concepts*. Oxford University Press.
3. Bothe et al. (2021). Stitch anti-unification.
4. Spelke & Kinzler (2007). Core knowledge. *Developmental Science*.
5. Ellis et al. (2021). DreamCoder. *PLDI*.
6. Chollet (2019). ARC-AGI benchmark.
7. Lake et al. (2017). Building machines that learn and think like people.
8. Ellis et al. (2023). LAPS.
9. Grand et al. (2024). LILO.
