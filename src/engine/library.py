"""
library.py — Growing library of typed combinators.

The Library is the central data structure of the wake-sleep loop.
It starts as the Spelke-initialized DSL and grows via compression
(abstraction sleep phase). Each new abstraction is a named, typed,
composable combinator.

MDL objective: total_cost = L(Library) + Σᵢ L(program_i | Library)
Accept new combinator if total cost decreases.
"""

from __future__ import annotations
import json
import math
import hashlib
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Type, Arrow,
)
from src.engine.program import Program, ProgramNode


@dataclass
class Abstraction:
    """
    A learned abstraction — a new combinator discovered during sleep.

    This is the DreamCoder concept of an invented primitive:
    a recurring fragment across solved programs, promoted to first-class
    named function in the library.

    The Carey analogue: a placeholder whose content is the compressed body.
    """
    name: str
    type_signature: Type
    body: ProgramNode              # The AST of the abstraction body
    source_programs: list[str]     # Task IDs where this pattern was found
    systems_composed: set[str]     # Which Spelke systems are combined
    reuse_count: int = 0           # How often it's used after invention
    invention_cycle: int = 0       # Which wake-sleep cycle invented it
    documentation: str = ""        # LLM-generated description (AutoDoc)
    log_probability: float = 0.0
    mdl_savings: float = 0.0      # How much description length it saves

    @property
    def is_cross_system(self) -> bool:
        """The Carey signature: does this abstraction bridge Spelke systems?"""
        core = {"OBJECTS", "FORMS", "NUMBER", "AGENTS", "PERSONS", "PLACES"}
        return len(self.systems_composed & core) > 1

    def to_primitive(self, implementation: Callable) -> Primitive:
        """Convert to a Primitive for the registry."""
        return Primitive(
            name=self.name,
            type_signature=self.type_signature,
            implementation=implementation,
            system=SpelkeSystem.GLUE,  # Invented abstractions are cross-system glue
            description=self.documentation,
            log_probability=self.log_probability,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": str(self.type_signature),
            "systems_composed": sorted(self.systems_composed),
            "is_cross_system": self.is_cross_system,
            "reuse_count": self.reuse_count,
            "invention_cycle": self.invention_cycle,
            "documentation": self.documentation,
            "mdl_savings": self.mdl_savings,
            "source_count": len(self.source_programs),
        }


class Library:
    """
    The growing library of typed combinators.

    Lifecycle:
    1. Initialized with Spelke sub-libraries (L₀)
    2. Wake phase: solve tasks using L
    3. Abstraction sleep: refactor recurring patterns → new abstractions
    4. Dreaming sleep: sample programs from updated prior → train recognition
    5. Iterate: L grows deeper each cycle

    This mirrors DreamCoder's library exactly, but with Spelke initialization.
    """

    def __init__(self, registry: PrimitiveRegistry):
        self.base_registry = registry
        self.abstractions: list[Abstraction] = []
        self._cycle = 0
        self._history: list[dict] = []  # Snapshot at each cycle

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def total_size(self) -> int:
        """Total number of available combinators (base + invented)."""
        return len(self.base_registry) + len(self.abstractions)

    @property
    def base_size(self) -> int:
        return len(self.base_registry)

    @property
    def invented_size(self) -> int:
        return len(self.abstractions)

    def all_primitive_names(self) -> list[str]:
        """All available primitive names (base + invented)."""
        names = self.base_registry.names()
        names.extend(a.name for a in self.abstractions)
        return names

    def get_primitive(self, name: str) -> Optional[Primitive]:
        """Look up any primitive (base or invented)."""
        if name in self.base_registry:
            return self.base_registry[name]
        for a in self.abstractions:
            if a.name == name:
                # For evaluation we need the full primitive
                return Primitive(
                    name=a.name,
                    type_signature=a.type_signature,
                    implementation=lambda *args: None,  # Placeholder
                    system=SpelkeSystem.GLUE,
                    description=a.documentation,
                    log_probability=a.log_probability,
                )
        return None

    def add_abstraction(self, abstraction: Abstraction) -> bool:
        """
        Add a new abstraction to the library if it passes MDL check.

        Returns True if accepted, False if rejected.
        MDL criterion: accept if total description length decreases.
        """
        if abstraction.mdl_savings <= 0:
            return False

        abstraction.invention_cycle = self._cycle
        self.abstractions.append(abstraction)
        return True

    def increment_cycle(self) -> None:
        """Record cycle completion and snapshot library state."""
        snapshot = {
            "cycle": self._cycle,
            "base_size": self.base_size,
            "invented_size": self.invented_size,
            "total_size": self.total_size,
            "abstractions": [a.to_dict() for a in self.abstractions],
            "cross_system_count": sum(1 for a in self.abstractions if a.is_cross_system),
        }
        self._history.append(snapshot)
        self._cycle += 1

    def cross_system_abstractions(self) -> list[Abstraction]:
        """Abstractions that bridge multiple Spelke systems — the Carey signature."""
        return [a for a in self.abstractions if a.is_cross_system]

    def abstractions_by_cycle(self) -> dict[int, list[Abstraction]]:
        """Group abstractions by the cycle in which they were invented."""
        by_cycle: dict[int, list[Abstraction]] = {}
        for a in self.abstractions:
            by_cycle.setdefault(a.invention_cycle, []).append(a)
        return by_cycle

    def most_reused(self, top_k: int = 10) -> list[Abstraction]:
        """Most frequently reused abstractions."""
        return sorted(self.abstractions, key=lambda a: -a.reuse_count)[:top_k]

    def pcfg_weights(self) -> dict[str, float]:
        """PCFG prior weights for all combinators."""
        weights = self.base_registry.to_pcfg_weights()
        for a in self.abstractions:
            weights[a.name] = a.log_probability
        return weights

    def description_length_of_library(self) -> float:
        """L(Library) — cost of encoding the library itself."""
        base_cost = len(self.base_registry) * 1.0  # Each base primitive costs 1 unit
        abstraction_cost = sum(
            a.body.size() if a.body else 1.0
            for a in self.abstractions
        )
        return base_cost + abstraction_cost

    def save(self, path: str | Path) -> None:
        """Serialize library to JSON."""
        data = {
            "cycle": self._cycle,
            "base_registry": self.base_registry.manifest(),
            "abstractions": [a.to_dict() for a in self.abstractions],
            "history": self._history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> str:
        lines = [
            f"Library (cycle {self._cycle}):",
            f"  Base primitives: {self.base_size}",
            f"  Invented abstractions: {self.invented_size}",
            f"  Total combinators: {self.total_size}",
        ]
        cs = self.cross_system_abstractions()
        if cs:
            lines.append(f"  Cross-system abstractions (Carey signature): {len(cs)}")
            for a in cs[:5]:
                lines.append(f"    {a.name}: {sorted(a.systems_composed)}")
        if self.abstractions:
            top = self.most_reused(3)
            lines.append(f"  Most reused:")
            for a in top:
                lines.append(f"    {a.name}: used {a.reuse_count}x")
        return "\n".join(lines)
