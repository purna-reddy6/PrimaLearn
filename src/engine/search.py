"""
search.py — Program search over the Spelke primitive library.

Two strategies:
1. HEURISTIC SOLVER: Directly applies Spelke primitives using task analysis
   heuristics (symmetry detection, object manipulation patterns, etc.)
   This runs fast and gets the first wave of solved tasks.

2. ENUMERATIVE SEARCH: Systematic enumeration of typed programs.
   Bounded by timeout and max attempts. Uses the PCFG prior for ordering.

The heuristic solver is the practical engine for initial experiments.
The enumerative search is the principled DreamCoder-compatible backbone.
"""

from __future__ import annotations
import time
import signal
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class _StrategyTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _StrategyTimeout("Strategy timed out")
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, Type, Arrow,
)
from src.engine.program import (
    Program, ProgramNode, PrimNode, AppNode, LamNode, VarNode, LitNode,
)
from src.engine.library import Library
from src.arc.grid import ArcTask, Grid


@dataclass
class SearchResult:
    """Result of searching for a program that solves a task."""
    program: Optional[Program] = None
    solved: bool = False
    attempts: int = 0
    time_seconds: float = 0.0
    candidates_explored: int = 0
    best_partial_score: float = 0.0
    strategy: str = ""


# ──────────────────────────────────────────────────────────────────────
# Heuristic Solver — fast, pattern-matching approach
# ──────────────────────────────────────────────────────────────────────

class HeuristicSolver:
    """
    Applies Spelke primitives using task-analysis heuristics.

    For each ARC task, analyzes the training examples to hypothesize
    a transformation rule, then verifies it on all training pairs.

    This is not blind enumeration — it uses the *structure* of the
    Spelke modules to guide search. The modules tell us *what to look for*.
    """

    def __init__(self, library: Library):
        self.library = library
        self.reg = library.base_registry

    def solve(self, task: ArcTask) -> SearchResult:
        start = time.time()
        result = SearchResult()

        # Get training pairs as numpy arrays
        pairs = [(ex.input.data, ex.output.data) for ex in task.train]

        # Try each strategy in priority order
        from src.engine.strategies import ALL_EXTENDED_STRATEGIES
        from src.engine.strategies_v2 import ALL_EXTENDED_STRATEGIES_V2
        from src.engine.compose import ALL_COMPOSITIONAL_STRATEGIES
        from src.engine.strategies_v3 import ALL_EXTENDED_STRATEGIES_V3
        
        strategies = [
            ("identity", self._try_identity),
            ("single_transform", self._try_single_transforms),
            ("color_replace", self._try_color_replacement),
            ("object_recolor", self._try_object_recolor),
            ("geometric_transform", self._try_geometric_transforms),
            ("symmetry_completion", self._try_symmetry_completion),
            ("scaling", self._try_scaling),
            ("tiling", self._try_tiling),
            ("object_filter", self._try_object_filter),
            ("object_sort", self._try_object_sort),
            ("crop_object", self._try_crop_to_object),
            ("border", self._try_border_ops),
            ("overlay", self._try_overlay),
            ("analogy", self._try_analogy),
        ]
        # Add all extended strategies (v1 + v2 + v3 + compositional)
        all_ext = (ALL_EXTENDED_STRATEGIES + ALL_EXTENDED_STRATEGIES_V2 +
                   ALL_EXTENDED_STRATEGIES_V3 + ALL_COMPOSITIONAL_STRATEGIES)
        for name, fn in all_ext:
            strategies.append((name, lambda pairs, f=fn: f(pairs)))

        for name, strategy_fn in strategies:
            result.attempts += 1
            result.candidates_explored += 1
            try:
                # Per-strategy timeout: 2 seconds max
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(2)
                try:
                    prog_fn = strategy_fn(pairs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                if prog_fn is not None:
                    # Verify on ALL training pairs
                    if self._verify(prog_fn, pairs):
                        result.solved = True
                        result.strategy = name
                        result.program = Program(
                            root=LitNode(f"<{name}>"),
                            task_id=task.task_id,
                            source="heuristic",
                        )
                        # Store the actual callable for evaluation
                        result.program._callable = prog_fn
                        result.time_seconds = time.time() - start
                        return result
            except (_StrategyTimeout, Exception):
                signal.alarm(0)
                continue

        result.time_seconds = time.time() - start
        return result

    def _verify(self, fn: Callable, pairs: list[tuple[np.ndarray, np.ndarray]]) -> bool:
        """Verify a candidate function on all training pairs."""
        for inp, expected in pairs:
            try:
                output = fn(inp)
                if output is None:
                    return False
                if not isinstance(output, np.ndarray):
                    return False
                if output.shape != expected.shape:
                    return False
                if not np.array_equal(output, expected):
                    return False
            except Exception:
                return False
        return True

    # ── Strategy: Identity ──
    def _try_identity(self, pairs):
        def identity(g):
            return g.copy()
        if self._verify(identity, pairs):
            return identity
        return None

    # ── Strategy: Single transforms (rotate, flip, transpose) ──
    def _try_single_transforms(self, pairs):
        transforms = [
            lambda g: np.rot90(g, k=-1).copy(),      # rotate90
            lambda g: np.rot90(g, k=2).copy(),        # rotate180
            lambda g: np.rot90(g, k=-3).copy(),       # rotate270
            lambda g: np.fliplr(g).copy(),             # flip_h
            lambda g: np.flipud(g).copy(),             # flip_v
            lambda g: g.T.copy(),                      # transpose
            lambda g: np.fliplr(g.T).copy(),           # anti-transpose
        ]
        for t in transforms:
            if self._verify(t, pairs):
                return t
        return None

    # ── Strategy: Color replacement ──
    def _try_color_replacement(self, pairs):
        # Detect if it's a simple color swap
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        diff_mask = inp0 != out0
        if not diff_mask.any():
            return None

        # Build color mapping from first pair
        color_map = {}
        for r in range(inp0.shape[0]):
            for c in range(inp0.shape[1]):
                ic, oc = int(inp0[r, c]), int(out0[r, c])
                if ic != oc:
                    if ic in color_map and color_map[ic] != oc:
                        return None  # Inconsistent mapping
                    color_map[ic] = oc

        if not color_map:
            return None

        def apply_map(g):
            result = g.copy()
            for old_c, new_c in color_map.items():
                result[g == old_c] = new_c
            return result

        if self._verify(apply_map, pairs):
            return apply_map
        return None

    # ── Strategy: Object recoloring based on properties ──
    def _try_object_recolor(self, pairs):
        from src.spelke_dsl.l_objects import _extract_objects

        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        in_objs = _extract_objects(inp0)
        out_objs = _extract_objects(out0)

        if len(in_objs) != len(out_objs):
            return None
        if not in_objs:
            return None

        # Check if objects stay in place but change color
        # Match by position
        recolor_rules = []
        for io in in_objs:
            matched = None
            for oo in out_objs:
                if io.cells == oo.cells:
                    matched = oo
                    break
            if matched is None:
                return None
            if io.color != matched.color:
                recolor_rules.append((io.size, io.color, matched.color))

        if not recolor_rules:
            return None

        # Check: recolor by size?
        # Find if there's a consistent rule
        # Try: smallest gets one color, largest gets another
        sizes = sorted(set(r[0] for r in recolor_rules))
        size_to_new_color = {}
        for s, old_c, new_c in recolor_rules:
            if s in size_to_new_color and size_to_new_color[s] != new_c:
                size_to_new_color = None
                break
            size_to_new_color[s] = new_c

        if size_to_new_color:
            def recolor_by_size(g):
                objs = _extract_objects(g)
                result = g.copy()
                for obj in objs:
                    if obj.size in size_to_new_color:
                        for r, c in obj.cells:
                            result[r, c] = size_to_new_color[obj.size]
                return result
            if self._verify(recolor_by_size, pairs):
                return recolor_by_size

        return None

    # ── Strategy: Geometric transforms on input ──
    def _try_geometric_transforms(self, pairs):
        # Try compositions of two transforms
        single_transforms = [
            lambda g: np.rot90(g, k=-1).copy(),
            lambda g: np.rot90(g, k=2).copy(),
            lambda g: np.fliplr(g).copy(),
            lambda g: np.flipud(g).copy(),
            lambda g: g.T.copy(),
        ]

        # Try: concat input with transform of input
        for t in single_transforms:
            # Horizontal concat
            def make_hconcat(transform):
                def fn(g):
                    return np.hstack([g, transform(g)])
                return fn

            def make_vconcat(transform):
                def fn(g):
                    return np.vstack([g, transform(g)])
                return fn

            hc = make_hconcat(t)
            if self._verify(hc, pairs):
                return hc

            vc = make_vconcat(t)
            if self._verify(vc, pairs):
                return vc

        return None

    # ── Strategy: Symmetry completion ──
    def _try_symmetry_completion(self, pairs):
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        # Check if output is input with added horizontal symmetry
        def complete_h_sym(g):
            result = g.copy()
            h, w = g.shape
            for r in range(h):
                for c in range(w):
                    mirror_c = w - 1 - c
                    if g[r, c] == 0 and g[r, mirror_c] != 0:
                        result[r, c] = g[r, mirror_c]
                    elif g[r, c] != 0 and g[r, mirror_c] == 0:
                        result[r, mirror_c] = g[r, c]
            return result

        def complete_v_sym(g):
            result = g.copy()
            h, w = g.shape
            for r in range(h):
                for c in range(w):
                    mirror_r = h - 1 - r
                    if g[r, c] == 0 and g[mirror_r, c] != 0:
                        result[r, c] = g[mirror_r, c]
                    elif g[r, c] != 0 and g[mirror_r, c] == 0:
                        result[mirror_r, c] = g[r, c]
            return result

        for fn in [complete_h_sym, complete_v_sym]:
            if self._verify(fn, pairs):
                return fn

        return None

    # ── Strategy: Scaling ──
    def _try_scaling(self, pairs):
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape

        if oh == 0 or ow == 0 or ih == 0 or iw == 0:
            return None

        # Check for integer scaling
        if oh % ih == 0 and ow % iw == 0:
            sr, sc = oh // ih, ow // iw
            if sr == sc and sr > 1:
                factor = sr
                def scale_up(g):
                    return np.repeat(np.repeat(g, factor, axis=0), factor, axis=1)
                if self._verify(scale_up, pairs):
                    return scale_up

        if ih % oh == 0 and iw % ow == 0:
            sr, sc = ih // oh, iw // ow
            if sr == sc and sr > 1:
                factor = sr
                def scale_down(g):
                    return g[::factor, ::factor].copy()
                if self._verify(scale_down, pairs):
                    return scale_down

        return None

    # ── Strategy: Tiling ──
    def _try_tiling(self, pairs):
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape

        if ih == 0 or iw == 0:
            return None

        if oh % ih == 0 and ow % iw == 0:
            nr, nc = oh // ih, ow // iw
            if nr >= 1 and nc >= 1 and (nr > 1 or nc > 1):
                def tile(g):
                    return np.tile(g, (nr, nc))
                if self._verify(tile, pairs):
                    return tile

        return None

    # ── Strategy: Filter objects ──
    def _try_object_filter(self, pairs):
        from src.spelke_dsl.l_objects import _extract_objects

        inp0, out0 = pairs[0]
        in_objs = _extract_objects(inp0)

        if len(in_objs) < 2:
            return None

        # Try: output is just the largest object
        largest = max(in_objs, key=lambda o: o.size)
        largest_grid = largest.to_grid()
        if largest_grid.shape == out0.shape and np.array_equal(largest_grid, out0):
            def get_largest(g):
                objs = _extract_objects(g)
                if not objs:
                    return g
                big = max(objs, key=lambda o: o.size)
                return big.to_grid()
            if self._verify(get_largest, pairs):
                return get_largest

        # Try: output is the smallest object
        smallest = min(in_objs, key=lambda o: o.size)
        smallest_grid = smallest.to_grid()
        if smallest_grid.shape == out0.shape and np.array_equal(smallest_grid, out0):
            def get_smallest(g):
                objs = _extract_objects(g)
                if not objs:
                    return g
                sm = min(objs, key=lambda o: o.size)
                return sm.to_grid()
            if self._verify(get_smallest, pairs):
                return get_smallest

        # Try: output is object of specific color
        out_objs = _extract_objects(out0)
        if len(out_objs) == 1:
            target_color = out_objs[0].color
            for io in in_objs:
                if io.color == target_color:
                    io_grid = io.to_grid()
                    if io_grid.shape == out0.shape and np.array_equal(io_grid, out0):
                        def get_by_color(g, tc=target_color):
                            objs = _extract_objects(g)
                            matches = [o for o in objs if o.color == tc]
                            if matches:
                                return matches[0].to_grid()
                            return g
                        if self._verify(get_by_color, pairs):
                            return get_by_color

        return None

    # ── Strategy: Sort objects ──
    def _try_object_sort(self, pairs):
        # Check if output rearranges objects by some property
        return None  # Complex — skip for pilot

    # ── Strategy: Crop to object ──
    def _try_crop_to_object(self, pairs):
        from src.spelke_dsl.l_objects import _extract_objects

        inp0, out0 = pairs[0]
        in_objs = _extract_objects(inp0)

        for obj in in_objs:
            r0, c0, r1, c1 = obj.bbox
            cropped = inp0[r0:r1+1, c0:c1+1].copy()
            if cropped.shape == out0.shape and np.array_equal(cropped, out0):
                # Crop to this object's bbox
                target_color = obj.color
                def crop_to_obj(g, tc=target_color):
                    objs = _extract_objects(g)
                    matches = [o for o in objs if o.color == tc]
                    if matches:
                        o = matches[0]
                        r0, c0, r1, c1 = o.bbox
                        return g[r0:r1+1, c0:c1+1].copy()
                    return g
                if self._verify(crop_to_obj, pairs):
                    return crop_to_obj

        return None

    # ── Strategy: Border operations ──
    def _try_border_ops(self, pairs):
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape

        # Check if output has 1-pixel border added
        if oh == ih + 2 and ow == iw + 2:
            # Check what color the border is
            border_color = int(out0[0, 0])
            inner = out0[1:-1, 1:-1]
            if np.array_equal(inner, inp0):
                def add_border(g, bc=border_color):
                    h, w = g.shape
                    result = np.full((h+2, w+2), bc, dtype=g.dtype)
                    result[1:-1, 1:-1] = g
                    return result
                if self._verify(add_border, pairs):
                    return add_border

        # Check if output has border removed
        if oh == ih - 2 and ow == iw - 2 and ih > 2 and iw > 2:
            inner = inp0[1:-1, 1:-1]
            if np.array_equal(inner, out0):
                def remove_border(g):
                    return g[1:-1, 1:-1].copy()
                if self._verify(remove_border, pairs):
                    return remove_border

        return None

    # ── Strategy: Grid overlay ──
    def _try_overlay(self, pairs):
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        from src.spelke_dsl.l_objects import _extract_objects

        # Check if output is overlay of objects in specific order
        in_objs = _extract_objects(inp0)
        if len(in_objs) < 2:
            return None

        # Try: overlay with rotated/flipped version of self
        transforms = [
            lambda g: np.fliplr(g),
            lambda g: np.flipud(g),
            lambda g: np.rot90(g, k=-1) if g.shape[0] == g.shape[1] else g,
            lambda g: np.rot90(g, k=2),
        ]

        for t in transforms:
            def make_overlay(transform):
                def fn(g):
                    transformed = transform(g)
                    if transformed.shape != g.shape:
                        return g
                    result = g.copy()
                    mask = transformed != 0
                    result[mask] = transformed[mask]
                    return result
                return fn

            fn = make_overlay(t)
            if self._verify(fn, pairs):
                return fn

        return None

    # ── Strategy: Analogical transfer ──
    def _try_analogy(self, pairs):
        if len(pairs) < 2:
            return None

        # Only attempt analogy on small grids to avoid O(n²) blowup
        for inp, _ in pairs:
            if inp.shape[0] > 15 or inp.shape[1] > 15:
                return None

        from src.spelke_dsl.analogy import _analogical_transfer

        # Use first pair as base, try transfer to remaining
        base_in, base_out = pairs[0]

        def analogy_fn(g, bi=base_in, bo=base_out):
            return _analogical_transfer(bi, bo, g)

        if self._verify(analogy_fn, pairs):
            return analogy_fn

        return None


# ──────────────────────────────────────────────────────────────────────
# SpelkeSolver — main solver integrating heuristic + enumeration
# ──────────────────────────────────────────────────────────────────────

class SpelkeSolver:
    """
    ARC solver using the Spelke-initialized library.

    Uses the heuristic solver for fast pattern matching,
    with enumerative search as fallback.
    """

    def __init__(self, library: Library, max_attempts: int = 1000, timeout: float = 60.0):
        self.library = library
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.heuristic = HeuristicSolver(library)

    def solve(self, task: ArcTask, max_attempts: Optional[int] = None) -> Any:
        """Solve a single ARC task."""
        from src.arc.evaluator import TaskResult

        start = time.time()

        # Phase 1: Heuristic solver (fast)
        h_result = self.heuristic.solve(task)

        if h_result.solved and h_result.program:
            # Compute test predictions
            predicted = []
            gt = [ex.output.data for ex in task.test]
            prog_fn = h_result.program._callable

            for test_ex in task.test:
                try:
                    pred = prog_fn(test_ex.input.data)
                    predicted.append(pred if isinstance(pred, np.ndarray) else None)
                except Exception:
                    predicted.append(None)

            all_correct = all(
                p is not None and g is not None
                and p.shape == g.shape and np.array_equal(p, g)
                for p, g in zip(predicted, gt)
            )

            return TaskResult(
                task_id=task.task_id,
                solved=all_correct,
                predicted_outputs=predicted,
                ground_truth=gt,
                search_attempts=h_result.attempts,
                solve_time_seconds=time.time() - start,
                program_found=f"heuristic:{h_result.strategy}",
            )

        # Phase 2: Enumerative search would go here
        # (skipped for now — too slow without recognition network)

        return TaskResult(
            task_id=task.task_id,
            solved=False,
            predicted_outputs=[None] * len(task.test),
            ground_truth=[ex.output.data for ex in task.test],
            search_attempts=h_result.attempts,
            solve_time_seconds=time.time() - start,
        )

    @property
    def name(self) -> str:
        return "SpelkeSolver"
