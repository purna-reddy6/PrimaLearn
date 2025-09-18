"""
wake_sleep.py — Full DreamCoder-style wake-sleep loop.

The real learning system:
  WAKE:             Recognition → Heuristic → AST Solver → Enumerator
  ABSTRACTION SLEEP: Stitch anti-unification → register as callable primitives
  DREAMING SLEEP:   Dream generator → train neural recognition network
  ITERATE:          Library grows, recognition improves, enumeration goes deeper
"""

from __future__ import annotations
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from src.spelke_dsl import build_spelke_library, PrimitiveRegistry
from src.engine.library import Library, Abstraction
from src.engine.program import Program, LitNode
from src.engine.search import SpelkeSolver
from src.engine.ast_solver import ASTSolver
from src.engine.compression import CompressionEngine
from src.arc.grid import ArcTask
from src.arc.evaluator import TaskResult, EvalResults

logger = logging.getLogger(__name__)


@dataclass
class WakeSleepConfig:
    n_iterations: int = 10
    search_timeout_per_task: float = 60.0
    max_search_attempts: int = 1000
    max_program_depth: int = 4
    max_program_size: int = 20
    min_abstraction_frequency: int = 2
    min_abstraction_size: int = 2
    max_abstractions_per_cycle: int = 15
    dream_samples_per_cycle: int = 200
    enumeration_budget: float = 5.0   # seconds per task for enumerator
    enumeration_max_cost: int = 4     # max AST cost for enumeration
    use_enumerator: bool = True       # whether to run real enumeration
    use_rust_enumerator: bool = False  # use Rust binary instead of Python TypeDirectedEnumerator
    use_neural_recognition: bool = True
    checkpoint_dir: Optional[str] = None
    verbose: bool = True


@dataclass
class CycleResult:
    cycle: int
    n_tasks: int
    n_solved: int
    solve_rate: float
    n_heuristic: int
    n_ast: int
    n_enumerated: int
    new_abstractions: list[dict]
    cross_system_abstractions: int
    total_library_size: int
    cycle_time_seconds: float
    solved_task_ids: list[str]
    strategies_used: dict = field(default_factory=dict)
    recognition_loss: float = 0.0
    # Compounding evidence: tasks newly solved THIS cycle using invented abstractions
    n_abstraction_compounded: int = 0
    abstraction_compounded_tasks: list[str] = field(default_factory=list)
    # Tasks newly solved this cycle (not carried over from prior cycles)
    newly_solved_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "n_tasks": self.n_tasks,
            "n_solved": self.n_solved,
            "solve_rate": self.solve_rate,
            "n_heuristic": self.n_heuristic,
            "n_ast": self.n_ast,
            "n_enumerated": self.n_enumerated,
            "new_abstractions": self.new_abstractions,
            "cross_system_abstractions": self.cross_system_abstractions,
            "total_library_size": self.total_library_size,
            "cycle_time_seconds": self.cycle_time_seconds,
            "strategies_used": self.strategies_used,
            "recognition_loss": self.recognition_loss,
            "n_abstraction_compounded": self.n_abstraction_compounded,
            "abstraction_compounded_tasks": self.abstraction_compounded_tasks,
            "newly_solved_task_ids": self.newly_solved_task_ids,
        }


class WakeSleepEngine:
    """
    Full DreamCoder-style wake-sleep engine with:
    - Type-directed enumeration (real program synthesis)
    - Stitch anti-unification compression
    - Neural recognition network (numpy MLP)
    - Dream generation for recognition training
    """

    def __init__(self, config: WakeSleepConfig):
        self.config = config
        self.library: Optional[Library] = None
        self.cycle_results: list[CycleResult] = []
        self._solved_programs: dict[str, Program] = {}
        self._alternative_programs: list[Program] = []  # extra corpus for compression
        self._recognition = None
        self._enumerator = None

    def initialize(self, registry: Optional[PrimitiveRegistry] = None) -> None:
        if registry is None:
            registry = build_spelke_library()
        self.library = Library(registry)

        # Initialize neural recognition network
        if self.config.use_neural_recognition:
            try:
                from src.engine.recognition import NeuralRecognitionNet
                self._recognition = NeuralRecognitionNet()
                self._recognition.initialize(list(registry._primitives.keys()))
                if self.config.verbose:
                    logger.info(f"Neural recognition initialized: "
                                f"{self._recognition.W1.shape} -> {self._recognition.W2.shape}")
            except Exception as e:
                logger.warning(f"Neural recognition init failed: {e}. Using heuristic fallback.")
                self._recognition = None

        # Initialize enumerator
        if self.config.use_enumerator:
            try:
                if self.config.use_rust_enumerator:
                    from src.engine.rust_enumerator import RustEnumerator
                    self._enumerator = RustEnumerator(
                        max_cost=self.config.enumeration_max_cost,
                        time_budget=self.config.enumeration_budget,
                    )
                    if not self._enumerator.is_available():
                        logger.warning("Rust enumerator binary not found — falling back to Python")
                        from src.engine.enumerator import TypeDirectedEnumerator
                        self._enumerator = TypeDirectedEnumerator(
                            max_cost=self.config.enumeration_max_cost,
                            time_budget=self.config.enumeration_budget,
                        )
                    elif self.config.verbose:
                        logger.info(f"Rust enumerator initialized: max_cost={self.config.enumeration_max_cost}")
                else:
                    from src.engine.enumerator import TypeDirectedEnumerator
                    self._enumerator = TypeDirectedEnumerator(
                        max_cost=self.config.enumeration_max_cost,
                        time_budget=self.config.enumeration_budget,
                    )
                    if self.config.verbose:
                        logger.info(f"Enumerator initialized: max_cost={self.config.enumeration_max_cost}")
            except Exception as e:
                logger.warning(f"Enumerator init failed: {e}. Skipping enumeration.")
                self._enumerator = None

        if self.config.verbose:
            logger.info("Initialized library:")
            logger.info(self.library.summary())

    def run(self, tasks: list[ArcTask]) -> list[CycleResult]:
        if self.library is None:
            self.initialize()

        results = []
        prior_solved_ids: set[str] = set()
        for iteration in range(self.config.n_iterations):
            if self.config.verbose:
                print(f"\n{'='*70}")
                print(f"  CYCLE {iteration + 1}/{self.config.n_iterations}")
                print(f"  Library size: {self.library.total_size}")
                print(f"{'='*70}")

            cycle_start = time.time()

            # ── WAKE ──
            n_heuristic, n_ast, n_enumerated, strategy_counts, compound_info = self._wake(tasks, cycle=iteration)

            # ── ABSTRACTION SLEEP ──
            new_abstractions = self._abstraction_sleep()

            # ── DREAMING SLEEP ──
            recognition_loss = self._dreaming_sleep(tasks)

            # ── RECORD ──
            cycle_time = time.time() - cycle_start
            solved_ids = list(self._solved_programs.keys())
            newly_solved = [tid for tid in solved_ids if tid not in prior_solved_ids]
            prior_solved_ids = set(solved_ids)

            cycle_result = CycleResult(
                cycle=iteration,
                n_tasks=len(tasks),
                n_solved=len(solved_ids),
                solve_rate=len(solved_ids) / len(tasks) if tasks else 0,
                n_heuristic=n_heuristic,
                n_ast=n_ast,
                n_enumerated=n_enumerated,
                new_abstractions=[a.to_dict() for a in new_abstractions],
                cross_system_abstractions=len(self.library.cross_system_abstractions()),
                total_library_size=self.library.total_size,
                cycle_time_seconds=cycle_time,
                solved_task_ids=solved_ids,
                strategies_used=strategy_counts,
                recognition_loss=recognition_loss,
                n_abstraction_compounded=compound_info["n_compounded"],
                abstraction_compounded_tasks=compound_info["compounded_tasks"],
                newly_solved_task_ids=newly_solved,
            )
            results.append(cycle_result)
            self.cycle_results.append(cycle_result)
            self.library.increment_cycle()

            if self.config.verbose:
                print(f"\n  Results:")
                print(f"    Solved: {cycle_result.n_solved}/{cycle_result.n_tasks} "
                      f"({cycle_result.solve_rate:.1%})")
                print(f"    By method: heuristic={n_heuristic}, ast={n_ast}, enumerated={n_enumerated}")
                print(f"    Newly solved this cycle: {len(newly_solved)}")
                if compound_info["n_compounded"] > 0:
                    print(f"    ★ COMPOUNDING: {compound_info['n_compounded']} new tasks solved via "
                          f"invented abstractions: {compound_info['compounded_tasks']}")
                print(f"    New abstractions: {len(new_abstractions)}")
                for a in new_abstractions:
                    cs_flag = "✦ CROSS-SYSTEM" if a.is_cross_system else ""
                    print(f"      {a.name}: savings={a.mdl_savings:.1f} reuse={a.reuse_count} {cs_flag}")
                print(f"    Cross-system total: {cycle_result.cross_system_abstractions}")
                print(f"    Library size: {cycle_result.total_library_size}")
                if recognition_loss > 0:
                    print(f"    Recognition loss: {recognition_loss:.4f}")
                print(f"    Time: {cycle_time:.1f}s")

            if self.config.checkpoint_dir:
                self._checkpoint(iteration)

        return results

    def _wake(self, tasks: list[ArcTask], cycle: int = 0) -> tuple[int, int, int, dict, dict]:
        """
        Wake phase: solve tasks using all available methods in priority order.
        Returns (n_heuristic, n_ast, n_enumerated, strategy_counts, compound_info)
        compound_info: {"n_compounded": int, "compounded_tasks": list[str]}
          — tasks newly solved THIS cycle using at least one invented abstraction (abs_*).
        """
        if self.config.verbose:
            print(f"\n  [WAKE] Solving {len(tasks)} tasks...")

        ast_solver = ASTSolver(self.library)
        heuristic = SpelkeSolver(
            self.library,
            max_attempts=self.config.max_search_attempts,
            timeout=self.config.search_timeout_per_task,
        )

        n_heuristic = 0
        n_ast = 0
        n_enumerated = 0
        strategy_counts = {}
        newly_solved = 0
        compounded_tasks: list[str] = []  # tasks solved via invented abstractions this cycle

        # Count existing solved by method
        for prog in self._solved_programs.values():
            if prog.source == "heuristic":
                n_heuristic += 1
            elif prog.source == "ast_solver":
                n_ast += 1
            elif prog.source == "enumerator":
                n_enumerated += 1

        # BUG 2 FIX: Determine exact resume index from mid-wake checkpoint
        resume_from = 0
        if self.config.checkpoint_dir:
            ckpt_dir = Path(self.config.checkpoint_dir)
            mid_ckpt_path = ckpt_dir / f"mid_wake_cycle{cycle}.json"
            if mid_ckpt_path.exists():
                try:
                    with open(mid_ckpt_path) as f:
                        mid_ckpt = json.load(f)
                    resume_from = mid_ckpt.get("resume_from_task_index", 0)
                    if resume_from > 0 and self.config.verbose:
                        print(f"    [RESUME] Resuming from task index {resume_from}", flush=True)
                except Exception:
                    pass

        # BUG 1 FIX: Open per-run JSONL task log for real-time tail -f visibility
        task_log_file = None
        if self.config.checkpoint_dir:
            ckpt_dir = Path(self.config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            task_log_file = open(ckpt_dir / f"task_log_cycle{cycle}.jsonl", "a")

        try:
            for i, task in enumerate(tasks):
                # BUG 2 FIX: Skip tasks already processed in a prior (partial) wake
                if i < resume_from:
                    continue

                if task.task_id in self._solved_programs:
                    # BUG 1 FIX: Log skipped tasks too so the log is contiguous
                    print(f"[TASK {i+1}/{len(tasks)}] {task.task_id} skip (already solved)", flush=True)
                    continue

                # BUG 1 FIX: Log task start with timestamp
                t0 = time.time()
                print(f"[TASK {i+1}/{len(tasks)}] {task.task_id} start", flush=True)

                # Get recognition priors if available
                priors = {}
                if self._recognition is not None and self._recognition._initialized:
                    try:
                        priors = self._recognition.predict_primitive_priors(task)
                    except Exception:
                        pass

                solved = False
                method = "unsolved"

                # Phase 1: AST solver (fast, produces compressible programs)
                try:
                    ast_prog = ast_solver.solve(task)
                    if ast_prog is not None:
                        self._solved_programs[task.task_id] = ast_prog
                        n_ast += 1
                        newly_solved += 1
                        prims = ast_prog.primitives_used()
                        for prim in prims:
                            strategy_counts[prim] = strategy_counts.get(prim, 0) + 1
                        # Compounding check: did this new solve use an invented abstraction?
                        if any(p.startswith("abs_") for p in prims):
                            compounded_tasks.append(task.task_id)
                        solved = True
                        method = "ast"
                        if self.config.verbose:
                            print(f"      ✓ {task.task_id} [ast] (+{newly_solved} this cycle)", flush=True)
                        # Also try enumerator for an alternative deeper program
                        # to enrich the compression corpus
                        if self._enumerator is not None:
                            try:
                                alt = self._enumerator.enumerate(task, self.library, priors)
                                if alt is not None and alt.root.to_str() != ast_prog.root.to_str():
                                    alt.source = "enumerator_alt"
                                    self._alternative_programs.append(alt)
                            except Exception:
                                pass
                except Exception:
                    pass

                # Phase 2: Type-directed enumerator (real program synthesis)
                if not solved and self._enumerator is not None:
                    try:
                        enum_prog = self._enumerator.enumerate(task, self.library, priors)
                        if enum_prog is not None:
                            self._solved_programs[task.task_id] = enum_prog
                            n_enumerated += 1
                            newly_solved += 1
                            prims = enum_prog.primitives_used()
                            for prim in prims:
                                strategy_counts[prim] = strategy_counts.get(prim, 0) + 1
                            if any(p.startswith("abs_") for p in prims):
                                compounded_tasks.append(task.task_id)
                            solved = True
                            method = "enum"
                            if self.config.verbose:
                                print(f"      ✓ {task.task_id} [enum] (+{newly_solved} this cycle)", flush=True)
                    except Exception as e:
                        logger.debug(f"Enumerator error on {task.task_id}: {e}")

                # Phase 3: Heuristic solver (pattern matching fallback)
                if not solved:
                    try:
                        h_result = heuristic.solve(task)
                        if h_result.solved and h_result.program_found:
                            self._solved_programs[task.task_id] = Program(
                                root=LitNode(h_result.program_found),
                                task_id=task.task_id,
                                source="heuristic",
                            )
                            n_heuristic += 1
                            newly_solved += 1
                            sname = h_result.program_found.replace("heuristic:", "")
                            strategy_counts[f"h:{sname}"] = strategy_counts.get(f"h:{sname}", 0) + 1
                            solved = True
                            method = f"heuristic:{sname}"
                            if self.config.verbose:
                                print(f"      ✓ {task.task_id} [heuristic:{sname}] (+{newly_solved} this cycle)", flush=True)
                    except Exception:
                        pass

                # BUG 1 FIX: Log task completion with elapsed time and method
                elapsed = time.time() - t0
                print(f"[TASK {i+1}/{len(tasks)}] {task.task_id} done {elapsed:.1f}s [{method}]", flush=True)
                if task_log_file is not None:
                    task_log_file.write(json.dumps({
                        "task_id": task.task_id,
                        "index": i,
                        "elapsed_s": round(elapsed, 3),
                        "method": method,
                        "cycle": cycle,
                    }) + "\n")
                    task_log_file.flush()

                # BUG 2 FIX: Save mid-wake checkpoint after EVERY task with exact resume index
                if self.config.checkpoint_dir:
                    ckpt_dir = Path(self.config.checkpoint_dir)
                    mid_ckpt = {
                        "cycle": cycle,
                        "tasks_done": i + 1,
                        "tasks_total": len(tasks),
                        "resume_from_task_index": i + 1,
                        "solved_ids": list(self._solved_programs.keys()),
                        "n_ast": n_ast,
                        "n_enumerated": n_enumerated,
                        "n_heuristic": n_heuristic,
                    }
                    with open(ckpt_dir / f"mid_wake_cycle{cycle}.json", "w") as f:
                        json.dump(mid_ckpt, f)

                if (i + 1) % 50 == 0 and self.config.verbose:
                    print(f"    [{i+1}/{len(tasks)}] Total solved: {len(self._solved_programs)} "
                          f"(+{newly_solved} this cycle) | "
                          f"ast={n_ast} enum={n_enumerated} heuristic={n_heuristic}", flush=True)
        finally:
            if task_log_file is not None:
                task_log_file.close()

        if self.config.verbose:
            total = len(self._solved_programs)
            print(f"    Wake done: {total} total solved (+{newly_solved} new)")
            print(f"    AST: {n_ast} | Enumerated: {n_enumerated} | Heuristic: {n_heuristic}")
            if compounded_tasks:
                print(f"    ★ {len(compounded_tasks)} new tasks solved via invented abstractions")

        compound_info = {"n_compounded": len(compounded_tasks), "compounded_tasks": compounded_tasks}
        return n_heuristic, n_ast, n_enumerated, strategy_counts, compound_info

    def _abstraction_sleep(self) -> list[Abstraction]:
        """
        Abstraction sleep: compress AST + enumerated programs with Stitch.
        Register accepted abstractions as callable primitives.
        """
        # Only use primary solved programs for compression — enumerator_alt programs
        # pollute the corpus: they solve the same tasks via different AST paths,
        # inflating FORMS pattern frequency while OBJECTS subtrees stay rare,
        # causing the top-K savings cutoff to exclude OBJECTS abstractions entirely.
        compressible = [
            p for p in self._solved_programs.values()
            if p.source in ("ast_solver", "enumerator", "enumerator_rewritten")
            and p.root is not None
        ]

        # Also track heuristic programs for frequency analysis (their strategy
        # names imply which primitives are conceptually used, even though
        # their ASTs are opaque LitNodes)
        heuristic_progs = [
            p for p in self._solved_programs.values()
            if p.source == "heuristic"
        ]

        if self.config.verbose:
            print(f"\n  [ABSTRACTION SLEEP] Compressing {len(compressible)} programs...")

        if len(compressible) < 2:
            if self.config.verbose:
                print(f"    Need ≥2 AST programs for compression.")
            return []

        from src.engine.stitch import StitchCompressor

        compressor = StitchCompressor(
            self.library,
            min_frequency=self.config.min_abstraction_frequency,
            min_size=self.config.min_abstraction_size,
            max_abstractions=self.config.max_abstractions_per_cycle,
        )

        new_abstractions = compressor.compress(compressible)

        accepted = []
        for abs_ in new_abstractions:
            if self.library.add_abstraction(abs_):
                accepted.append(abs_)

        # Register accepted abstractions as real callable primitives
        if accepted:
            try:
                compressor.register_abstractions_as_primitives(
                    accepted, self.library.base_registry
                )
                if self.config.verbose:
                    print(f"    Registered {len(accepted)} abstractions as callable primitives")
            except Exception as e:
                logger.warning(f"Abstraction registration failed: {e}")

        # Extend recognition network output layer to include new abstractions
        if accepted and self._recognition is not None:
            try:
                all_names = list(self.library.base_registry._primitives.keys())
                self._recognition.extend_primitives(all_names)
            except Exception as e:
                logger.warning(f"Recognition extension failed: {e}")

        # Rewrite corpus using new abstractions (makes programs shorter)
        # We keep original programs in _solved_programs so compression in future
        # cycles always sees the full-depth ASTs. Rewritten versions are stored
        # separately for recognition training only.
        if accepted and compressible:
            try:
                rewritten = compressor.rewrite_corpus(compressible, accepted)
                # Store rewritten as alternatives for recognition — do NOT
                # overwrite _solved_programs so compression keeps seeing deep originals
                for prog in rewritten:
                    if prog.task_id and prog.source == "enumerator_rewritten":
                        self._alternative_programs.append(prog)
                if self.config.verbose:
                    print(f"    Rewrote {len(rewritten)} programs with new abstractions")
            except Exception as e:
                logger.warning(f"Corpus rewriting failed: {e}")

        if self.config.verbose:
            print(f"    Accepted {len(accepted)}/{len(new_abstractions)} abstractions")
            for a in accepted:
                if a.is_cross_system:
                    print(f"    ✦ CAREY SIGNATURE: {a.name} bridges {a.systems_composed}")

        return accepted

    def _dreaming_sleep(self, tasks: list[ArcTask]) -> float:
        """
        Dreaming sleep: generate synthetic tasks, train neural recognition.
        Returns final training loss (0 if recognition not used).
        """
        if self.config.verbose:
            print(f"\n  [DREAMING SLEEP] Generating dreams + training recognition...")

        recognition_loss = 0.0

        # Generate dream tasks
        dream_tasks = []
        if self.config.use_neural_recognition and self._recognition is not None:
            try:
                from src.engine.dream_generator import DreamGenerator
                gen = DreamGenerator(
                    self.library,
                    n_dreams=self.config.dream_samples_per_cycle,
                )
                dream_tasks = gen.generate()
                if self.config.verbose:
                    print(f"    Generated {len(dream_tasks)} dream tasks")
            except Exception as e:
                logger.warning(f"Dream generation failed: {e}")

        # Build solved task list for recognition training
        # v2: Include heuristic programs too by mapping strategy names to
        # the primitives they conceptually use.  This means the recognition
        # network learns from ALL 46 solved tasks, not just the ~25 with
        # real ASTs.
        _HEURISTIC_PRIM_MAP = {
            "identity": [],
            "single_transform": ["rotate90", "flip_h", "flip_v", "transpose"],
            "color_replace": ["replace_color"],
            "object_recolor": ["extract_objects", "obj_recolor", "render_objects_on"],
            "geometric_transform": ["rotate90", "flip_h", "flip_v"],
            "symmetry_completion": ["sym_horizontal", "sym_vertical"],
            "scaling": ["scale_up", "scale_down"],
            "tiling": ["tile"],
            "object_filter": ["extract_objects", "obj_largest", "obj_smallest", "render_object"],
            "object_sort": ["extract_objects", "obj_sort_size", "render_objects"],
            "crop_object": ["extract_objects", "crop"],
            "border": ["make_border"],
            "overlay": ["overlay", "flip_h", "flip_v"],
            "analogy": ["analogy_transfer"],
        }

        solved_for_training = []
        for tid, prog in self._solved_programs.items():
            matching = [t for t in tasks if t.task_id == tid]
            if not matching:
                continue
            if prog.root is None:
                continue
            if prog.source in ("ast_solver", "enumerator"):
                try:
                    prims_used = list(prog.primitives_used())
                except Exception:
                    prims_used = []
                solved_for_training.append((matching[0], prims_used))
            elif prog.source == "heuristic":
                # Map strategy name to primitive set
                try:
                    strategy_str = prog.root.to_str()
                except Exception:
                    strategy_str = ""
                strategy_name = strategy_str.strip("<>").replace("heuristic:", "")
                inferred_prims = []
                for key, plist in _HEURISTIC_PRIM_MAP.items():
                    if key in strategy_name:
                        inferred_prims = plist
                        break
                if not inferred_prims:
                    # Fallback: extract common prims from strategy name
                    inferred_prims = ["extract_objects", "render_objects"]
                solved_for_training.append((matching[0], inferred_prims))

        # Train recognition network
        if self._recognition is not None and self._recognition._initialized:
            if dream_tasks or solved_for_training:
                try:
                    recognition_loss = self._recognition.train(
                        dream_tasks=dream_tasks,
                        solved_tasks=solved_for_training,
                        n_epochs=20,
                        lr=0.01,
                        momentum=0.9,
                    )
                    # Update enumerator recognition priors in library
                    if self.config.verbose:
                        print(f"    Recognition trained on {len(solved_for_training)} solved "
                              f"+ {len(dream_tasks)} dreams | loss={recognition_loss:.4f}")
                except Exception as e:
                    logger.warning(f"Recognition training failed: {e}")
            else:
                if self.config.verbose:
                    print(f"    No training data yet (need solved tasks first)")
        else:
            # Fallback: record strategy outcomes for heuristic recognition
            for tid, prog in self._solved_programs.items():
                strategy = prog.source
                if hasattr(self, '_heuristic_recognition'):
                    self._heuristic_recognition.record_outcome(strategy, True)

        return recognition_loss

    def _checkpoint(self, cycle: int) -> None:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.library.save(ckpt_dir / f"library_cycle_{cycle}.json")

        results_data = [r.to_dict() for r in self.cycle_results]
        with open(ckpt_dir / "cycle_results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        programs_data = {
            tid: prog.to_dict() for tid, prog in self._solved_programs.items()
        }
        with open(ckpt_dir / f"programs_cycle_{cycle}.json", "w") as f:
            json.dump(programs_data, f, indent=2)

        # Save recognition network weights for full state resumption
        if self._recognition is not None and self._recognition._initialized:
            try:
                weights_path = str(ckpt_dir / f"recognition_cycle_{cycle}.npz")
                self._recognition.save_weights(weights_path)
                if self.config.verbose:
                    logger.info(f"Saved recognition weights to {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to save recognition weights: {e}")

        # Save final results.json
        if self.cycle_results:
            last = self.cycle_results[-1]
            with open(ckpt_dir / "results.json", "w") as f:
                json.dump({
                    "solved": last.n_solved,
                    "total": last.n_tasks,
                    "solve_rate": last.solve_rate,
                    "library_size": last.total_library_size,
                    "cycles": [r.to_dict() for r in self.cycle_results],
                }, f, indent=2)

    def load_checkpoint(self, checkpoint_dir: str, cycle: int) -> None:
        """
        Restore full engine state from a saved checkpoint so a run can resume
        from cycle `cycle + 1` rather than restarting from scratch.

        Restores:
          - Library abstractions (re-registered as callable primitives)
          - Solved programs (so wake phase skips already-solved tasks)
          - Recognition network weights (full numpy state)
          - Cycle results history (so summary includes all prior cycles)
          - Cycle counter advanced to `cycle + 1`

        This enables true mid-run resumption: if an experiment runs 3 cycles
        cleanly but a bug is found at cycle 4, you can fix the bug and
        resume from the cycle 3 checkpoint instead of restarting from cycle 0.

        Usage:
          engine.initialize(registry)
          engine.load_checkpoint('experiments/outputs/run_X/spelke', cycle=2)
          # engine.run(tasks) will now start from cycle 3
        """
        import json
        from pathlib import Path
        from src.engine.stitch import StitchCompressor

        ckpt = Path(checkpoint_dir)
        lib_path = ckpt / f"library_cycle_{cycle}.json"
        progs_path = ckpt / f"programs_cycle_{cycle}.json"
        recognition_path = ckpt / f"recognition_cycle_{cycle}.npz"
        cycle_results_path = ckpt / "cycle_results.json"

        if not lib_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {lib_path}")

        # ── Restore library abstractions ──────────────────────────────
        with open(lib_path) as f:
            lib_data = json.load(f)

        from src.engine.library import Abstraction
        from src.spelke_dsl.base import Arrow, TypeVariable
        from src.spelke_dsl.base import tgrid

        for a_dict in lib_data.get("abstractions", []):
            abs_ = Abstraction(
                name=a_dict["name"],
                type_signature=Arrow(TypeVariable("x"), tgrid),  # simplified
                body=None,
                source_programs=[],
                systems_composed=set(a_dict.get("systems_composed", [])),
                reuse_count=a_dict.get("reuse_count", 0),
                mdl_savings=a_dict.get("mdl_savings", 1.0),
                documentation=a_dict.get("documentation", ""),
            )
            abs_.invention_cycle = a_dict.get("invention_cycle", 0)
            if abs_.name not in [x.name for x in self.library.abstractions]:
                self.library.abstractions.append(abs_)

        # Re-register as callable primitives so the enumerator can use them
        try:
            compressor = StitchCompressor(self.library)
            accepted = [a for a in self.library.abstractions]
            compressor.register_abstractions_as_primitives(
                accepted, self.library.base_registry
            )
        except Exception as e:
            logger.warning(f"Checkpoint abstraction re-registration failed: {e}")

        # ── Restore solved programs ───────────────────────────────────
        if progs_path.exists():
            with open(progs_path) as f:
                progs_data = json.load(f)
            from src.engine.program import Program
            for task_id, prog_dict in progs_data.items():
                if task_id not in self._solved_programs:
                    # Create a stub Program so wake phase skips this task
                    stub = Program(
                        root=None,
                        task_id=task_id,
                        source=prog_dict.get("source", "checkpoint"),
                    )
                    self._solved_programs[task_id] = stub

        # ── Restore recognition network weights ───────────────────────
        if self._recognition is not None:
            if recognition_path.exists():
                loaded = self._recognition.load_weights(str(recognition_path))
                if loaded and self.config.verbose:
                    print(f"  Recognition weights restored from {recognition_path}")
            else:
                # Fallback: at least extend primitives for new abstractions
                all_names = list(self.library.base_registry._primitives.keys())
                self._recognition.extend_primitives(all_names)

        # ── Restore cycle results history ─────────────────────────────
        if cycle_results_path.exists():
            try:
                with open(cycle_results_path) as f:
                    results_data = json.load(f)
                for r_dict in results_data:
                    cr = CycleResult(
                        cycle=r_dict["cycle"],
                        n_tasks=r_dict["n_tasks"],
                        n_solved=r_dict["n_solved"],
                        solve_rate=r_dict["solve_rate"],
                        n_heuristic=r_dict["n_heuristic"],
                        n_ast=r_dict["n_ast"],
                        n_enumerated=r_dict["n_enumerated"],
                        new_abstractions=r_dict.get("new_abstractions", []),
                        cross_system_abstractions=r_dict.get("cross_system_abstractions", 0),
                        total_library_size=r_dict.get("total_library_size", 0),
                        cycle_time_seconds=r_dict.get("cycle_time_seconds", 0),
                        solved_task_ids=r_dict.get("solved_task_ids", []),
                        strategies_used=r_dict.get("strategies_used", {}),
                        recognition_loss=r_dict.get("recognition_loss", 0),
                        n_abstraction_compounded=r_dict.get("n_abstraction_compounded", 0),
                        abstraction_compounded_tasks=r_dict.get("abstraction_compounded_tasks", []),
                        newly_solved_task_ids=r_dict.get("newly_solved_task_ids", []),
                    )
                    self.cycle_results.append(cr)
            except Exception as e:
                logger.warning(f"Failed to restore cycle results: {e}")

        # ── Advance cycle counter ─────────────────────────────────────
        self.library._cycle = cycle + 1

        if self.config.verbose:
            n_abs = len(self.library.abstractions)
            n_solved = len(self._solved_programs)
            n_history = len(self.cycle_results)
            print(f"  Checkpoint loaded: cycle={cycle}, "
                  f"abstractions={n_abs}, solved={n_solved} tasks, "
                  f"history={n_history} prior cycles restored")

    def summary(self) -> str:
        lines = ["Wake-Sleep Results:"]
        for r in self.cycle_results:
            compound_str = (f" ★COMPOUND={r.n_abstraction_compounded}"
                           if r.n_abstraction_compounded > 0 else "")
            lines.append(
                f"  Cycle {r.cycle}: {r.n_solved}/{r.n_tasks} solved "
                f"({r.solve_rate:.1%}), lib={r.total_library_size}, "
                f"cross={r.cross_system_abstractions}, "
                f"[h={r.n_heuristic} ast={r.n_ast} enum={r.n_enumerated}]{compound_str} "
                f"time={r.cycle_time_seconds:.1f}s"
            )
        # Compounding summary
        total_compounded = sum(r.n_abstraction_compounded for r in self.cycle_results)
        if total_compounded > 0:
            lines.append(f"\n  COMPOUNDING: {total_compounded} total tasks solved via "
                        f"invented abstractions across all cycles")
        if self.library:
            lines.append(f"\n{self.library.summary()}")
        return "\n".join(lines)
