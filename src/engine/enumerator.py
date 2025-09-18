"""
enumerator.py — Type-directed bottom-up program enumerator.

Implements cost-bounded enumeration over the Spelke DSL to find Grid→Grid
programs that solve ARC tasks. Uses a bottom-up table indexed by
(type_repr, cost) for efficient search.

Architecture: DreamCoder-style wake phase — enumerate programs, evaluate on
training pairs, return the first complete solution found.

v2 — MULTI-TYPE SEARCH:
  The table now populates intermediate types (list[object], object, int, bool)
  so programs can flow through non-grid types and back to grid.  This unlocks
  ~119 primitives that were previously unreachable (extract_objects, obj_largest,
  render_objects, count_objects, etc.).  Final candidate evaluation still targets
  tgrid — intermediate types are stepping stones, not solutions.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

import numpy as np

from src.spelke_dsl.base import (
    Arrow, ListType, PairType, Type, TypeConstructor, TypeVariable,
    tgrid, tcolor, tint, tbool, tobject, tlist,
)
from src.engine.program import (
    AppNode, LitNode, PrimNode, ProgramNode, VarNode, Program,
)
from src.engine.library import Library

logger = logging.getLogger(__name__)

# ── Per-type table caps to prevent memory explosion ──
# Intermediate types that can produce many entries are capped lower.
_TYPE_CAPS = {
    "grid":           500,
    "list[object]":   100,
    "object":         100,
    "int":             50,
    "color":           20,
    "bool":            20,
}
_DEFAULT_CAP = 200


def _type_repr(t: Type) -> str:
    return repr(t)


def _cap_for_type(t: Type) -> int:
    """Get the table cap for a given type."""
    name = repr(t)
    for key, cap in _TYPE_CAPS.items():
        if key in name:
            return cap
    return _DEFAULT_CAP


def _is_concrete(t: Type) -> bool:
    """Return True if this type contains no TypeVariables."""
    if isinstance(t, TypeVariable):
        return False
    if isinstance(t, TypeConstructor):
        return True
    if isinstance(t, Arrow):
        return _is_concrete(t.arg) and _is_concrete(t.result)
    if isinstance(t, ListType):
        return _is_concrete(t.element_type)
    if isinstance(t, PairType):
        return _is_concrete(t.first) and _is_concrete(t.second)
    # Assume concrete if we reach here
    return True


def _has_concrete_result(t: Type) -> bool:
    """Return True if the final return type is concrete."""
    while isinstance(t, Arrow):
        t = t.result
    return _is_concrete(t)


def _types_compatible(t1: Type, t2: Type) -> bool:
    """Two types are compatible if equal, or either is a TypeVariable."""
    if t1 == t2:
        return True
    if isinstance(t1, TypeVariable) or isinstance(t2, TypeVariable):
        return True
    return repr(t1) == repr(t2)


# Types that TypeVariable arguments should try (not ALL table types)
_TYPEVAR_CANDIDATE_TYPES = None  # lazily built


def _get_typevar_candidate_reprs() -> list[str]:
    """Types that we allow TypeVariable arguments to match against."""
    global _TYPEVAR_CANDIDATE_TYPES
    if _TYPEVAR_CANDIDATE_TYPES is None:
        _TYPEVAR_CANDIDATE_TYPES = [
            _type_repr(tgrid),
            _type_repr(tint),
            _type_repr(tcolor),
            _type_repr(tobject),
            _type_repr(tlist(tobject)),
        ]
    return _TYPEVAR_CANDIDATE_TYPES


class TypeDirectedEnumerator:
    """
    Bottom-up, cost-bounded enumerator over the Spelke DSL.

    Table: (type_repr, cost) → list[ProgramNode]

    Cost model:
      - VarNode("input") at type tgrid: cost 0
      - LitNode(color or int): cost 1
      - PrimNode: cost 1 for the primitive itself
      - AppNode: free (func_cost + arg_cost)

    v2: Multi-type search — the table populates intermediate types
    (list[object], object, int, bool) so programs can flow through
    non-grid types and back to grid.
    """

    def __init__(self, max_cost: int = 4, time_budget: float = 5.0):
        self.max_cost = max_cost
        self.time_budget = time_budget

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def enumerate(
        self,
        task,
        library: Library,
        primitive_priors: Optional[dict[str, float]] = None,
    ) -> Optional[Program]:
        """
        Find the first Grid→Grid program that solves all task training pairs.

        Uses a single SIGALRM for the entire task budget (2 syscalls total)
        to hard-kill pathological evaluate() calls that would otherwise hang.
        This replaces the old per-example SIGALRM which caused 20k+ syscalls.

        Returns:
            Program wrapping the solution, or None if not found within budget.
        """
        deadline = time.time() + self.time_budget
        priors = primitive_priors or {}
        extra_prims = self._make_analogy_prims(task, library)

        # Set SIGALRM BEFORE _build_table so the full enumerate() call —
        # including table construction — is covered by the timeout.
        # (Previously set after _build_table, leaving it unguarded.)
        on_main = threading.current_thread() is threading.main_thread()
        old_handler = None

        def _timeout_handler(_signum, _frame):
            raise TimeoutError("enumerate budget exceeded")

        if on_main:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max(1, int(self.time_budget)))

        table = self._build_table(library, priors, deadline, extra_prims)
        tgrid_repr = _type_repr(tgrid)

        result = None
        try:
            for cost in range(1, self.max_cost + 1):
                if time.time() > deadline:
                    logger.debug("Enumerator: time budget exhausted before cost %d", cost)
                    break
                candidates = table.get((tgrid_repr, cost), [])
                for node in candidates:
                    if time.time() > deadline:
                        break
                    if self._evaluate_on_task(node, task, deadline):
                        result = Program(
                            root=node,
                            task_id=task.task_id,
                            source="enumerator",
                        )
                        break
                if result is not None:
                    break
        except TimeoutError:
            logger.debug("Enumerator: SIGALRM timeout for task %s", task.task_id)
        finally:
            if on_main:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        # Explicitly clear the table so Python frees AppNodes incrementally
        # here rather than in a massive GC cascade at the next task boundary.
        table.clear()

        if result is None:
            logger.debug("No solution found for task %s", task.task_id)
        return result

    def _make_analogy_prims(self, task, library: Library) -> list:
        """
        Create task-conditioned analogy primitives by partially applying
        analogy_transfer to each training pair.

        analogy_transfer : tgrid → tgrid → tgrid → tgrid
        After partial application with (inp_example, out_example):
          analogy_from_pair_N : tgrid → tgrid

        This makes the Gentner analogy operator a first-class enumerable
        primitive without changing the type-directed search machinery.
        """
        from src.spelke_dsl.base import Primitive, SpelkeSystem
        prims = []
        base = library.base_registry
        if "analogy_transfer" not in base:
            return prims
        transfer_fn = base["analogy_transfer"].implementation
        for i, example in enumerate(task.train[:2]):   # use first 2 pairs max
            inp = example.input.data
            out = example.output.data
            # Partially apply: bind inp and out, leave new_input open
            partially_applied = transfer_fn(inp)(out)
            name = f"analogy_pair_{i}"
            prim = Primitive(
                name=name,
                type_signature=Arrow(tgrid, tgrid),
                implementation=partially_applied,
                system=SpelkeSystem.ANALOGY,
                description=f"Gentner analogy from training pair {i}",
                log_probability=-1.5,   # prior: slightly disfavoured vs base prims
            )
            prims.append(prim)
        return prims

    # ──────────────────────────────────────────────────────────────────
    # Table construction
    # ──────────────────────────────────────────────────────────────────

    def _build_table(
        self,
        library: Library,
        priors: dict[str, float],
        deadline: float,
        extra_prims: list | None = None,
    ) -> dict[tuple[str, int], list[ProgramNode]]:
        """
        Bottom-up table: (type_repr, cost) → list[ProgramNode].

        Fills cost levels 0 through max_cost in order.

        v2: Populates intermediate types (list[object], object, int, bool)
        so programs can flow through non-grid types.  Per-type caps prevent
        memory explosion on types with many producers.
        """
        registry = library.base_registry

        # Sort primitives: highest prior probability first.
        # Invented abstractions (names starting with "abs_") that are absent
        # from the recognition output get a warm default (0.3) so they are
        # tried before the long tail of low-prior base primitives, not after.
        all_prims = list(registry)
        if extra_prims:
            all_prims.extend(extra_prims)

        def _prior(p):
            if p.name in priors:
                return priors[p.name]
            if p.name.startswith("abs_"):
                return 0.3
            return 0.0

        all_prims.sort(key=lambda p: -_prior(p))

        # Use primitives whose final result type is concrete.
        # Allow TypeVariable in argument positions — invented abstractions
        # (e.g. 'x → grid) have polymorphic args but a known result type.
        usable_prims = [
            p for p in all_prims
            if isinstance(p.type_signature, Arrow)
            and _has_concrete_result(p.type_signature)
        ]

        table: dict[tuple[str, int], list[ProgramNode]] = {}

        # Per-type cap tracking
        type_caps: dict[str, int] = {}

        def _add(type_repr: str, cost: int, node: ProgramNode) -> None:
            key = (type_repr, cost)
            bucket = table.setdefault(key, [])
            cap = type_caps.get(type_repr)
            if cap is None:
                # Determine cap from type name
                for tname, tcap in _TYPE_CAPS.items():
                    if tname in type_repr:
                        cap = tcap
                        break
                if cap is None:
                    cap = _DEFAULT_CAP
                type_caps[type_repr] = cap
            if len(bucket) < cap:
                bucket.append(node)

        # ── Cost 0: input variable (type tgrid) ───────────────────────
        _add(_type_repr(tgrid), 0, VarNode("input"))

        # ── Cost 1: literal constants ──────────────────────────────────
        # Color literals 0..9
        for c in range(10):
            _add(_type_repr(tcolor), 1, LitNode(c))
        # Int literals 1..5
        for i in range(1, 6):
            _add(_type_repr(tint), 1, LitNode(i))

        # ── Costs 1..max_cost: primitive applications ──────────────────
        for target_cost in range(1, self.max_cost + 1):
            if time.time() > deadline:
                logger.debug("Table build: deadline hit at cost %d", target_cost)
                break

            # Shared counter caps total AppNodes explored across all _try_apply
            # recursion levels for this cost level. Prevents 125M+ candidate
            # explosions when cycle-2 library has ~200 primitives at depth 4.
            nodes_explored = [0]

            for prim in usable_prims:
                if time.time() > deadline:
                    break
                # Prim itself costs 1; remaining budget goes to arguments
                arg_budget = target_cost - 1
                if arg_budget < 0:
                    continue

                # Decompose curried Arrow into a list of (arg_type, result_type) steps
                steps: list[tuple[Type, Type]] = []
                cur: Type = prim.type_signature
                while isinstance(cur, Arrow):
                    steps.append((cur.arg, cur.result))
                    cur = cur.result

                if time.time() > deadline:
                    break
                prim_node = PrimNode(prim.name, prim)
                self._try_apply(
                    prim_node, steps, 0, arg_budget, target_cost, table, _add,
                    deadline, nodes_explored,
                )
                if nodes_explored[0] > 200_000:
                    logger.debug("Table build: node cap hit at cost %d", target_cost)
                    break

        return table

    def _try_apply(
        self,
        node: ProgramNode,
        steps: list[tuple[Type, Type]],
        step_idx: int,
        remaining: int,
        target_cost: int,
        table: dict,
        add_fn,
        deadline: float = float("inf"),
        nodes_explored: list[int] | None = None,
    ) -> None:
        """
        Recursively apply arguments to node by distributing `remaining` budget
        among the curried arg steps.

        When the last step is applied and the budget is exactly exhausted,
        record the resulting AppNode at target_cost with the result type.
        For intermediate steps (partial application), recurse.

        nodes_explored is a shared single-element list used as a mutable counter
        across all recursion levels. Callers can cap total work by checking it.
        """
        if step_idx >= len(steps):
            return

        if time.time() > deadline:
            return

        arg_type, result_type = steps[step_idx]
        is_last = (step_idx == len(steps) - 1)
        arg_type_repr = _type_repr(arg_type)

        for arg_cost in range(0, remaining + 1):
            if time.time() > deadline:
                return
            left = remaining - arg_cost

            # Gather sub-expressions of the right type at arg_cost
            candidates = table.get((arg_type_repr, arg_cost), [])

            # TypeVariable args: try a curated set of useful types
            # instead of scanning the entire table (prevents explosion)
            if isinstance(arg_type, TypeVariable):
                extra: list[ProgramNode] = []
                for tr in _get_typevar_candidate_reprs():
                    if tr != arg_type_repr:
                        extra.extend(table.get((tr, arg_cost), []))
                if extra:
                    candidates = list(candidates) + extra

            for arg_node in candidates:
                if time.time() > deadline:
                    return
                if nodes_explored is not None:
                    nodes_explored[0] += 1
                    if nodes_explored[0] > 200_000:
                        return
                app = AppNode(node, arg_node)

                if is_last:
                    # Must consume the entire budget here
                    if left == 0:
                        add_fn(_type_repr(result_type), target_cost, app)
                else:
                    # Partial application — continue with remaining steps
                    if left >= 0:
                        self._try_apply(
                            app,
                            steps,
                            step_idx + 1,
                            left,
                            target_cost,
                            table,
                            add_fn,
                            deadline,
                            nodes_explored,
                        )

    # ──────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────

    def _evaluate_on_task(self, node: ProgramNode, task, deadline: float = float("inf")) -> bool:
        """
        Return True iff node solves every training example exactly.
        """
        try:
            for example in task.train:
                if time.time() > deadline:
                    return False
                inp = example.input.data
                expected = example.output.data
                result = node.evaluate({"input": inp})
                if time.time() > deadline:
                    return False
                if not isinstance(result, np.ndarray):
                    return False
                if result.shape != expected.shape:
                    return False
                if not np.array_equal(result, expected):
                    return False
            return True
        except Exception:
            return False

    def _types_compatible(self, t1: Type, t2: Type) -> bool:
        """Public wrapper for type compatibility check."""
        return _types_compatible(t1, t2)

