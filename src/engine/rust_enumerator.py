"""
rust_enumerator.py — Python wrapper around the spelke-enumerator Rust binary.

Calls the Rust binary via subprocess with JSON stdin/stdout protocol.
Falls back to the Python TypeDirectedEnumerator if Rust binary is unavailable.

Architecture:
  - Python serializes library + task → JSON stdin
  - Rust enumerates programs and returns S-expressions as JSON stdout
  - Python deserializes S-expressions → ProgramNode objects → evaluates → returns solution

v1: Rust enumerates programs at type "grid"; Python evaluates each against task training pairs.
    This avoids implementing a full ARC grid evaluator in Rust (PyO3 FFI deferred to v2).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.spelke_dsl.base import Arrow
from src.engine.program import AppNode, LitNode, PrimNode, ProgramNode, VarNode, Program
from src.engine.library import Library

logger = logging.getLogger(__name__)

# Path to the compiled Rust binary (relative to bootstrap-substrate root)
_BINARY_PATH = Path(__file__).parent.parent.parent / "rust" / "spelke-enumerator" / "target" / "release" / "spelke-enumerator"


def _find_binary() -> Optional[Path]:
    """Find the compiled Rust binary."""
    if _BINARY_PATH.exists():
        return _BINARY_PATH
    return None


def _serialize_library(library: Library) -> list[dict]:
    """Serialize Library primitives for the Rust JSON interface."""
    from src.engine.enumerator import _has_concrete_result
    prims = []
    for p in library.base_registry:
        if not isinstance(p.type_signature, Arrow):
            continue
        if not _has_concrete_result(p.type_signature):
            continue
        prims.append({
            "name": p.name,
            "type_repr": repr(p.type_signature),
            "prior": getattr(p, "log_probability", 0.0),
        })
    return prims


def _parse_sexp(sexp: str, library: Library) -> Optional[ProgramNode]:
    """
    Parse an S-expression string back into a ProgramNode.

    S-expression format (matches Rust to_sexp()):
      "input"          → VarNode("input")
      "42"             → LitNode(42)
      "prim_name"      → PrimNode(prim_name, ...)
      "(f arg)"        → AppNode(f, arg)
      "((f a) b)"      → AppNode(AppNode(f, a), b)

    Returns None if parsing fails.
    """
    try:
        return _parse_sexp_inner(sexp.strip(), library)
    except Exception as e:
        logger.debug("Failed to parse sexp %r: %s", sexp, e)
        return None


def _parse_sexp_inner(s: str, library: Library) -> ProgramNode:
    """Inner recursive S-expression parser."""
    s = s.strip()

    if not s:
        raise ValueError("Empty s-expression")

    # Parenthesized application: (func arg)
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1].strip()
        # Find the split between func and arg (accounting for nested parens)
        func_str, arg_str = _split_app(inner)
        func = _parse_sexp_inner(func_str, library)
        arg = _parse_sexp_inner(arg_str, library)
        return AppNode(func, arg)

    # "input" variable
    if s == "input":
        return VarNode("input")

    # Integer literal
    try:
        v = int(s)
        return LitNode(v)
    except ValueError:
        pass

    # Primitive name — look up in library
    registry = library.base_registry
    prim = registry.get(s)
    if prim is not None:
        return PrimNode(s, prim)

    # Could be an abstraction (abs_N) not in base_registry — check abstractions
    if hasattr(library, 'abstractions'):
        for abs_entry in library.abstractions:
            if abs_entry.name == s:
                # Create a stub Prim for the abstraction
                # (will evaluate via the abstraction's implementation)
                from src.spelke_dsl.base import Primitive, SpelkeSystem
                stub = Primitive(
                    name=s,
                    type_signature=abs_entry.type_signature,
                    implementation=abs_entry.implementation,
                    system=SpelkeSystem.GLUE,
                    description=f"Abstraction {s}",
                )
                return PrimNode(s, stub)

    raise ValueError(f"Unknown primitive: {s!r}")


def _split_app(s: str) -> tuple[str, str]:
    """
    Split "(func arg)" inner content into func_str and arg_str.
    The split is at the first top-level space.
    Example: "(identity input)" inner = "identity input" → ("identity", "input")
    Example: "((f a) b)" inner = "(f a) b" → ("(f a)", "b")
    """
    depth = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ' ' and depth == 0:
            return s[:i].strip(), s[i + 1:].strip()
    raise ValueError(f"Cannot split application: {s!r}")


class RustEnumerator:
    """
    Drop-in replacement for TypeDirectedEnumerator that uses the Rust binary.

    Usage:
        enumerator = RustEnumerator(max_cost=5, time_budget=5.0)
        result = enumerator.enumerate(task, library)
    """

    def __init__(self, max_cost: int = 5, time_budget: float = 5.0):
        self.max_cost = max_cost
        self.time_budget = time_budget
        self._binary = _find_binary()
        if self._binary is None:
            logger.warning(
                "Rust binary not found at %s — falling back to Python enumerator",
                _BINARY_PATH
            )

    def is_available(self) -> bool:
        """Check whether the Rust binary is available."""
        return self._binary is not None

    def enumerate(
        self,
        task,
        library: Library,
        primitive_priors: Optional[dict[str, float]] = None,
    ) -> Optional[Program]:
        """
        Find the first Grid→Grid program that solves all task training pairs.

        Uses SIGALRM to hard-kill the entire call (Rust subprocess + Python evaluation)
        within the time budget. Matches TypeDirectedEnumerator's SIGALRM pattern.

        Falls back to Python TypeDirectedEnumerator if Rust binary unavailable.
        """
        if not self.is_available():
            # Fall back to Python
            from src.engine.enumerator import TypeDirectedEnumerator
            py_enum = TypeDirectedEnumerator(max_cost=self.max_cost, time_budget=self.time_budget)
            return py_enum.enumerate(task, library, primitive_priors)

        # Set SIGALRM to hard-kill the entire enumerate() call (Rust + evaluation)
        # This prevents pathological programs from hanging Python evaluation.
        on_main = threading.current_thread() is threading.main_thread()
        old_handler = None

        def _timeout_handler(_signum, _frame):
            raise TimeoutError("rust_enumerator budget exceeded")

        if on_main:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            # +2s over budget: Rust gets time_budget, evaluation gets 2s extra
            signal.alarm(max(1, int(self.time_budget) + 2))

        result = None
        try:
            result = self._enumerate_inner(task, library, primitive_priors)
        except TimeoutError:
            logger.debug("Rust enumerator: SIGALRM timeout for task %s", task.task_id)
        finally:
            if on_main:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        return result

    def _enumerate_inner(
        self,
        task,
        library: Library,
        primitive_priors: Optional[dict[str, float]] = None,
    ) -> Optional[Program]:
        """Inner implementation — wrapped by enumerate() with SIGALRM protection."""
        # Build JSON payload for Rust
        priors = primitive_priors or {}
        lib_prims = _serialize_library(library)

        # Override priors from primitive_priors dict
        for p in lib_prims:
            if p["name"] in priors:
                p["prior"] = priors[p["name"]]

        # Give Rust most of the budget; reserve 1s for Python evaluation
        rust_budget = max(0.5, self.time_budget - 1.0)

        payload = {
            "schema_version": 1,
            "library": lib_prims,
            "task_id": task.task_id,
            "max_cost": self.max_cost,
            "time_budget": rust_budget,
        }

        payload_bytes = json.dumps(payload).encode("utf-8")

        # Call Rust binary
        t0 = time.time()
        try:
            proc = subprocess.run(
                [str(self._binary)],
                input=payload_bytes,
                capture_output=True,
                timeout=rust_budget + 3.0,  # external hard timeout
            )
        except subprocess.TimeoutExpired:
            logger.debug("Rust enumerator subprocess timed out for task %s", task.task_id)
            return None
        except Exception as e:
            logger.error("Rust enumerator failed for task %s: %s", task.task_id, e)
            return None

        rust_elapsed = time.time() - t0

        if proc.returncode != 0:
            logger.error(
                "Rust enumerator returned code %d for task %s: %s",
                proc.returncode, task.task_id, proc.stderr.decode("utf-8", errors="replace")
            )
            return None

        # Parse output JSON
        try:
            output = json.loads(proc.stdout.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error("Rust enumerator: invalid JSON output for task %s: %s", task.task_id, e)
            return None

        programs = output.get("programs", [])
        n_explored = output.get("n_explored", 0)
        elapsed_ms = output.get("elapsed_ms", 0)
        result_status = output.get("result", "unknown")

        logger.debug(
            "Rust enumerator for task %s: status=%s programs=%d n_explored=%d elapsed_ms=%d rust_elapsed=%.2f",
            task.task_id, result_status, len(programs), n_explored, elapsed_ms, rust_elapsed
        )

        if not programs:
            return None

        # Evaluate each candidate program against the task.
        # deadline is relative to full time_budget (Rust got rust_budget, we have ~1s left)
        # SIGALRM protects against evaluation hangs.
        deadline = t0 + self.time_budget  # hard outer deadline
        eval_deadline = t0 + rust_elapsed + 1.0  # 1s for evaluation

        for entry in programs:
            if time.time() > eval_deadline or time.time() > deadline:
                break
            sexp = entry.get("sexp", "")
            node = _parse_sexp(sexp, library)
            if node is None:
                continue
            if self._evaluate_on_task(node, task, eval_deadline):
                return Program(
                    root=node,
                    task_id=task.task_id,
                    source="rust_enumerator",
                )

        return None

    def _evaluate_on_task(self, node: ProgramNode, task, deadline: float) -> bool:
        """Return True iff node solves every training example exactly."""
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


def build_rust_enumerator(max_cost: int = 5, time_budget: float = 5.0) -> RustEnumerator:
    """Factory function — convenience wrapper for RustEnumerator."""
    return RustEnumerator(max_cost=max_cost, time_budget=time_budget)
