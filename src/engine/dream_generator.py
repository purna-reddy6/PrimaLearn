"""
dream_generator.py — Synthetic dream task generator for the sleep phase.

Generates (program, input→output pairs) triples by sampling programs from
the library PCFG prior and executing them on random grids. These dreams
train the recognition network to guide future enumeration.

Architecture: DreamCoder sleep phase — sample programs, execute on random
grids, collect tasks that succeed without error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.spelke_dsl.base import Arrow, TypeVariable, tgrid
from src.engine.program import AppNode, PrimNode, ProgramNode, VarNode
from src.engine.library import Library

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# DreamTask dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DreamTask:
    """
    A synthetic (program, pairs) triple generated during the sleep phase.

    Fields:
        program_node:    The ProgramNode that was sampled and executed.
        primitive_names: All primitive names appearing in the program.
        systems_used:    Spelke system names touched by the program.
        pairs:           List of (input_array, output_array) numpy pairs.
        features:        Optional feature vector filled in later by recognition.
    """
    program_node: ProgramNode
    primitive_names: list[str]
    systems_used: list[str]
    pairs: list[tuple]          # list of (np.ndarray, np.ndarray)
    features: Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────────────────
# DreamGenerator
# ──────────────────────────────────────────────────────────────────────

class DreamGenerator:
    """
    Generates synthetic ARC-like tasks by sampling programs from the
    library PCFG prior and executing them on random grids.

    Sampling strategy:
      - 30%: multi-type pipeline dreams (extract_objects → transform → render)
      - 30%: cross-system dreams (combine primitives from different modules)
      - 40%: standard PCFG chains of Grid→Grid primitives

    v2: Pipeline dreams teach the recognition network about intermediate-type
    primitives (extract_objects, obj_largest, render_objects, count_objects)
    that were previously invisible to dreams.
    """

    # System pairs to explicitly dream about (Carey bootstrapping targets)
    _CROSS_SYSTEM_PAIRS = [
        ("OBJECTS", "NUMBER"),   # count objects → use count
        ("OBJECTS", "FORMS"),    # extract objects → apply geometry
        ("FORMS",   "NUMBER"),   # symmetry detection → count symmetric pairs
        ("OBJECTS", "ANALOGY"),  # object graph → structural transfer
    ]

    # Multi-type pipeline templates: (outer_prim, inner_prim) or
    # (outer, middle, inner) chains that go through non-grid types.
    # Each tuple is (prim_names...) where the last is applied to input.
    _PIPELINE_TEMPLATES = [
        # list[object] → grid pipelines
        ("render_objects", "extract_objects"),
        ("render_objects", "extract_objects_8conn"),
        ("render_objects", "obj_z_order", "extract_objects"),
        ("render_objects", "obj_sort_size", "extract_objects"),
        # object → grid pipelines
        ("render_object", "obj_largest", "extract_objects"),
        ("render_object", "obj_smallest", "extract_objects"),
        # render_objects_on(input, filter_color(extract_objects(input), C))
        # — can't template easily, handled via _sample_pipeline_program
    ]

    def __init__(self, library: Library, n_dreams: int = 200, seed: int = 42):
        self.library = library
        self.n_dreams = n_dreams
        self.rng = np.random.default_rng(seed)

        # Cache Grid→Grid primitives per system and overall
        self._gg_prims: list = []
        self._gg_weights: np.ndarray
        self._prims_by_system: dict[str, list] = {}

        self._cache_gg_primitives()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def generate(self) -> list[DreamTask]:
        """
        Generate up to n_dreams synthetic tasks.

        Returns the list of successful DreamTasks (may be shorter than
        n_dreams if many programs fail execution).
        """
        if not self._gg_prims:
            logger.debug("DreamGenerator: no Grid→Grid primitives found, returning empty")
            return []

        results: list[DreamTask] = []
        attempts = 0
        max_attempts = self.n_dreams * 5  # allow for failures

        while len(results) < self.n_dreams and attempts < max_attempts:
            attempts += 1

            # Dream type selection: 30% pipeline, 30% cross-system, 40% standard
            roll = self.rng.random()
            if roll < 0.30:
                node = self._sample_pipeline_program()
            elif roll < 0.60 and self._prims_by_system:
                node = self._sample_cross_system_program()
            else:
                node = self._sample_program()
            if node is None:
                continue

            # Generate 3 (input, output) pairs
            pairs: list[tuple] = []
            ok = True
            for _ in range(3):
                grid = self._random_grid(self.rng)
                try:
                    out = node.evaluate({"input": grid})
                    if not isinstance(out, np.ndarray):
                        ok = False
                        break
                    if out.ndim != 2:
                        ok = False
                        break
                    pairs.append((grid, out))
                except Exception:
                    ok = False
                    break

            if not ok or len(pairs) < 3:
                continue

            # Collect metadata
            prim_names = sorted(node.primitives_used())
            sys_names = sorted(
                s.name
                for s in node.systems_used(self.library.base_registry)
            )

            results.append(DreamTask(
                program_node=node,
                primitive_names=prim_names,
                systems_used=sys_names,
                pairs=pairs,
            ))

        logger.debug(
            "DreamGenerator: generated %d dreams from %d attempts "
            "(including pipeline and cross-system dreams)",
            len(results), attempts,
        )
        return results

    # ──────────────────────────────────────────────────────────────────
    # Program sampling
    # ──────────────────────────────────────────────────────────────────

    def _sample_pipeline_program(self) -> Optional[ProgramNode]:
        """
        Sample a program that goes through intermediate types.

        Uses pre-defined pipeline templates like:
          render_objects(extract_objects(input))
          render_object(obj_largest(extract_objects(input)))

        These dreams teach the recognition network that tasks with
        "multiple objects" features should predict extract_objects,
        obj_largest, render_objects, etc. — primitives the old
        grid→grid-only dreams never included.
        """
        registry = self.library.base_registry
        try:
            # Pick a random template
            template = self._PIPELINE_TEMPLATES[
                int(self.rng.integers(0, len(self._PIPELINE_TEMPLATES)))
            ]

            # Check all prims exist
            for pname in template:
                if pname not in registry:
                    return self._sample_program()  # fallback

            # Build the chain: template[0](template[1](...(template[-1](input))))
            inner: ProgramNode = VarNode("input")
            for pname in reversed(template):
                prim = registry[pname]
                inner = AppNode(PrimNode(pname, prim), inner)

            # Optionally add a grid→grid transform on top (50% of the time)
            if self.rng.random() < 0.5 and self._gg_prims:
                outer = self._gg_prims[
                    int(self.rng.integers(0, len(self._gg_prims)))
                ]
                inner = AppNode(PrimNode(outer.name, outer), inner)

            return inner
        except Exception:
            return None

    def _sample_cross_system_program(self) -> Optional[ProgramNode]:
        """
        Sample a program that deliberately combines primitives from two
        different Spelke systems. This teaches the recognition network
        that cross-system feature patterns predict cross-system primitives —
        the prerequisite for the enumerator to discover Carey abstractions.

        Strategy: pick a system pair, sample one primitive from each,
        compose them: outer_prim(inner_prim(input)).
        """
        try:
            # Pick a cross-system pair that has primitives in both systems
            available = [
                (s1, s2) for s1, s2 in self._CROSS_SYSTEM_PAIRS
                if s1 in self._prims_by_system and s2 in self._prims_by_system
                and self._prims_by_system[s1] and self._prims_by_system[s2]
            ]
            if not available:
                return self._sample_program()

            idx = int(self.rng.integers(0, len(available)))
            s1, s2 = available[idx]

            p1 = self._prims_by_system[s1][
                int(self.rng.integers(0, len(self._prims_by_system[s1])))
            ]
            p2 = self._prims_by_system[s2][
                int(self.rng.integers(0, len(self._prims_by_system[s2])))
            ]

            # depth 2: p1(p2(input))
            inner = AppNode(PrimNode(p2.name, p2), VarNode("input"))
            return AppNode(PrimNode(p1.name, p1), inner)
        except Exception:
            return None

    def _sample_program(self) -> Optional[ProgramNode]:
        """
        Sample a random Grid→Grid program from the library PCFG prior.

        Samples depth uniformly from {1, 2, 3} and chains that many
        Grid→Grid primitives together, each weighted by log_probability.

        Returns None if sampling fails.
        """
        if not self._gg_prims:
            return None

        try:
            depth = int(self.rng.integers(1, 4))  # 1, 2, or 3

            # Sample `depth` primitives with replacement, weighted by prior
            indices = self.rng.choice(
                len(self._gg_prims),
                size=depth,
                replace=True,
                p=self._gg_weights,
            )
            prims = [self._gg_prims[i] for i in indices]

            # Build the composed program: prim[0](prim[1](...(prim[d-1](input))))
            # Innermost: prim[d-1] applied to input var
            inner: ProgramNode = VarNode("input")
            for prim in reversed(prims):
                prim_node = PrimNode(prim.name, prim)
                inner = AppNode(prim_node, inner)

            return inner
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # Grid generation
    # ──────────────────────────────────────────────────────────────────

    def _random_grid(self, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a random ARC-like grid.

        Dimensions: 2–10 rows, 2–10 cols.
        Colors: drawn from 2–5 distinct colors (subset of ARC palette 0–9).
        Background (0) is always included to match ARC conventions.
        """
        h = int(rng.integers(2, 11))
        w = int(rng.integers(2, 11))
        n_colors = int(rng.integers(2, 6))

        # Always include 0 (background); pick n_colors-1 others from 1..9
        other_colors = rng.choice(9, size=n_colors - 1, replace=False) + 1
        palette = np.array([0] + list(other_colors), dtype=np.int8)

        # Sample each cell independently from the palette
        idx = rng.integers(0, len(palette), size=(h, w))
        grid = palette[idx].astype(np.int8)
        return grid

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _cache_gg_primitives(self) -> None:
        """
        Pre-filter and cache all Grid→Grid primitives from the base registry.

        A primitive qualifies if its type signature is Arrow(tgrid, tgrid),
        i.e. arg = tgrid and immediate result = tgrid (not another Arrow).
        We skip polymorphic arrows that contain TypeVariables in the
        Grid→Grid positions.
        """
        registry = self.library.base_registry
        prims = []
        for prim in registry:
            sig = prim.type_signature
            if not isinstance(sig, Arrow):
                continue
            # Must take a tgrid as first arg
            if isinstance(sig.arg, TypeVariable):
                continue
            if repr(sig.arg) != repr(tgrid):
                continue
            # Immediate result must be tgrid (not curried further)
            if repr(sig.result) != repr(tgrid):
                continue
            prims.append(prim)

        if not prims:
            logger.debug("DreamGenerator: no Grid→Grid primitives found in library")
            self._gg_prims = []
            self._gg_weights = np.array([], dtype=float)
            return

        self._gg_prims = prims

        # Build per-system index for cross-system dream sampling
        self._prims_by_system = {}
        for prim in prims:
            sys_name = prim.system.name if prim.system else "UNKNOWN"
            self._prims_by_system.setdefault(sys_name, []).append(prim)

        # Convert log_probabilities to proper probabilities via softmax-style
        log_probs = np.array([p.log_probability for p in prims], dtype=float)
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(len(prims), dtype=float)
            total = float(len(prims))
        self._gg_weights = probs / total

        logger.debug(
            "DreamGenerator: cached %d Grid→Grid primitives across %d systems",
            len(self._gg_prims), len(self._prims_by_system),
        )
