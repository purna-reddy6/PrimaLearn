"""
compression.py — Abstraction discovery via corpus compression.

Implements the "abstraction sleep" phase of the wake-sleep loop:
given a corpus of solved programs, identify recurring fragments
and propose them as new library abstractions.

Based on Stitch (Bowers et al. 2023, POPL) algorithm:
top-down corpus-guided search over partial abstractions.

MDL objective: total_cost = L(Library) + Σᵢ L(program_i | Library)
Accept new combinator if total cost decreases.
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Type, Arrow,
)
from src.engine.program import (
    Program, ProgramNode, PrimNode, AppNode, LamNode, VarNode, LitNode,
)
from src.engine.library import Library, Abstraction


@dataclass
class Fragment:
    """A recurring program fragment — candidate for abstraction."""
    pattern: ProgramNode
    occurrences: list[tuple[str, int]]  # (task_id, position) pairs
    frequency: int = 0
    mdl_savings: float = 0.0
    systems: set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return self.pattern.size()


class CompressionEngine:
    """
    Corpus-guided abstraction discovery.

    Algorithm (simplified Stitch):
    1. Collect all solved programs from the wake phase
    2. Extract all subtrees of each program
    3. Hash subtrees to find recurring fragments
    4. For each frequent fragment, compute MDL savings
    5. Accept fragments that decrease total description length
    6. Promote accepted fragments to new library abstractions
    """

    def __init__(
        self,
        library: Library,
        min_frequency: int = 2,
        min_size: int = 2,
        max_abstractions_per_cycle: int = 5,
    ):
        self.library = library
        self.min_frequency = min_frequency
        self.min_size = min_size
        self.max_abstractions = max_abstractions_per_cycle

    def compress(self, solved_programs: list[Program]) -> list[Abstraction]:
        """
        Run one round of abstraction discovery.

        1. Extract all subtrees from solved programs
        2. Find frequent recurring fragments
        3. Score by MDL savings
        4. Return top abstractions that decrease total cost
        """
        if not solved_programs:
            return []

        # Step 1: Extract all subtrees
        subtree_counts: Counter[str] = Counter()
        subtree_map: dict[str, ProgramNode] = {}
        subtree_sources: dict[str, list[str]] = defaultdict(list)

        for prog in solved_programs:
            subtrees = self._extract_subtrees(prog.root)
            for st in subtrees:
                key = st.to_str()
                subtree_counts[key] += 1
                subtree_map[key] = st
                if prog.task_id:
                    subtree_sources[key].append(prog.task_id)

        # Step 2: Filter by frequency and size
        candidates = []
        for key, count in subtree_counts.items():
            if count < self.min_frequency:
                continue
            node = subtree_map[key]
            if node.size() < self.min_size:
                continue

            # Compute MDL savings
            savings = self._compute_mdl_savings(
                node, count, len(solved_programs)
            )

            if savings > 0:
                fragment = Fragment(
                    pattern=node,
                    occurrences=[(tid, 0) for tid in subtree_sources.get(key, [])],
                    frequency=count,
                    mdl_savings=savings,
                    systems=self._get_systems(node),
                )
                candidates.append(fragment)

        # Step 3: Sort by MDL savings, take top
        candidates.sort(key=lambda f: -f.mdl_savings)
        top = candidates[:self.max_abstractions]

        # Step 4: Convert to Abstractions
        abstractions = []
        for i, frag in enumerate(top):
            name = f"abs_{self.library.cycle}_{i}"
            abs_ = Abstraction(
                name=name,
                type_signature=self._infer_fragment_type(frag.pattern),
                body=frag.pattern,
                source_programs=[t for t, _ in frag.occurrences],
                systems_composed=frag.systems,
                reuse_count=frag.frequency,
                mdl_savings=frag.mdl_savings,
            )
            abstractions.append(abs_)

        return abstractions

    def _extract_subtrees(self, node: ProgramNode) -> list[ProgramNode]:
        """Extract all subtrees (subexpressions) of a program."""
        result = [node]

        if isinstance(node, AppNode):
            result.extend(self._extract_subtrees(node.func))
            result.extend(self._extract_subtrees(node.arg))
        elif isinstance(node, LamNode):
            result.extend(self._extract_subtrees(node.body))
        elif isinstance(node, LetNode):
            result.extend(self._extract_subtrees(node.expr))
            result.extend(self._extract_subtrees(node.body))

        return result

    def _compute_mdl_savings(
        self,
        node: ProgramNode,
        frequency: int,
        corpus_size: int,
    ) -> float:
        """
        Compute MDL savings from abstracting this fragment.

        Savings = (frequency × fragment_size) - (fragment_size + frequency × 1)
                = fragment_size × (frequency - 1) - frequency

        The first term is the total encoding cost before abstraction.
        The second term is: cost of the abstraction definition + cost of
        references to it.
        """
        frag_size = node.size()
        reference_cost = 1.0  # Cost of referencing the abstraction

        cost_before = frequency * frag_size
        cost_after = frag_size + frequency * reference_cost  # definition + references

        return cost_before - cost_after

    def _get_systems(self, node: ProgramNode) -> set[str]:
        """
        Get transitive Spelke systems touched by a fragment.

        Checks both base primitives (direct) and any invented abstractions
        referenced in the fragment (transitive), so that a fragment reusing
        abs_0_2 (which bridges OBJECTS+NUMBER) is labelled correctly rather
        than inheriting no system tag.
        """
        systems = set()
        abs_systems: dict[str, set[str]] = {
            a.name: a.systems_composed
            for a in self.library.abstractions
        }
        for prim_name in node.primitives_used():
            if prim_name in self.library.base_registry:
                prim = self.library.base_registry[prim_name]
                systems.add(prim.system.name)
            elif prim_name in abs_systems:
                systems |= abs_systems[prim_name]
        return systems

    def _infer_fragment_type(self, node: ProgramNode) -> Type:
        """Infer the type of a fragment (simplified)."""
        if isinstance(node, PrimNode):
            return node.primitive.type_signature
        if isinstance(node, AppNode):
            func_type = self._infer_fragment_type(node.func)
            if isinstance(func_type, Arrow):
                return func_type.result
            return func_type
        if isinstance(node, LamNode):
            from src.spelke_dsl.base import TypeVariable
            body_type = self._infer_fragment_type(node.body)
            return Arrow(TypeVariable("x"), body_type)
        # Default
        from src.spelke_dsl.base import TypeVariable
        return TypeVariable("unknown")


class LetNode:
    """Stub for import compatibility."""
    pass
