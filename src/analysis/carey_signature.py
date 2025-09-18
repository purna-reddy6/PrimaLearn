"""
carey_signature.py — Detect cross-system integration in library growth.

The "Carey signature" is the hallmark of PrimaLearn:
new abstractions that integrate representations from distinct
core-knowledge systems. If the Spelke-initialized library produces
abstractions that span Objects+Number, or Forms+Objects, etc.,
that is direct evidence of Carey-style cross-system integration.

This module detects and quantifies such patterns.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Optional
from src.spelke_dsl.base import SpelkeSystem
from src.engine.library import Library, Abstraction


CORE_SYSTEMS = {
    SpelkeSystem.OBJECTS.name,
    SpelkeSystem.FORMS.name,
    SpelkeSystem.NUMBER.name,
    SpelkeSystem.AGENTS.name,
    SpelkeSystem.PERSONS.name,
    SpelkeSystem.PLACES.name,
}


@dataclass
class CareySignatureReport:
    """Analysis report on cross-system integration patterns."""
    total_abstractions: int = 0
    cross_system_count: int = 0
    cross_system_rate: float = 0.0
    system_pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    cross_system_abstractions: list[dict] = field(default_factory=list)
    growth_by_cycle: list[dict] = field(default_factory=list)
    integration_depth: float = 0.0  # Average number of systems per cross-system abs

    def summary(self) -> str:
        lines = [
            "Carey Signature Analysis",
            "=" * 40,
            f"Total abstractions: {self.total_abstractions}",
            f"Cross-system: {self.cross_system_count} ({self.cross_system_rate:.1%})",
            f"Integration depth: {self.integration_depth:.2f}",
        ]
        if self.system_pair_counts:
            lines.append("\nSystem pair frequencies:")
            for pair, count in sorted(self.system_pair_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {pair[0]} × {pair[1]}: {count}")
        if self.cross_system_abstractions:
            lines.append(f"\nCross-system abstractions:")
            for a in self.cross_system_abstractions[:10]:
                lines.append(f"  {a['name']}: {sorted(a['systems'])}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_abstractions": self.total_abstractions,
            "cross_system_count": self.cross_system_count,
            "cross_system_rate": self.cross_system_rate,
            "system_pair_counts": {f"{k[0]}_{k[1]}": v for k, v in self.system_pair_counts.items()},
            "integration_depth": self.integration_depth,
            "growth_by_cycle": self.growth_by_cycle,
        }


def analyze_carey_signature(library: Library) -> CareySignatureReport:
    """
    Analyze a library for Carey-style cross-system integration patterns.

    Looks for:
    1. Abstractions that compose primitives from >1 Spelke core system
    2. The specific system pairs that get integrated
    3. How cross-system integration grows across wake-sleep cycles
    4. Integration depth (how many systems per abstraction)
    """
    report = CareySignatureReport()
    report.total_abstractions = len(library.abstractions)

    if not library.abstractions:
        return report

    # Analyze each abstraction
    pair_counts: Counter[tuple[str, str]] = Counter()
    depths = []

    for abs_ in library.abstractions:
        core_systems = abs_.systems_composed & CORE_SYSTEMS
        if len(core_systems) > 1:
            report.cross_system_count += 1
            report.cross_system_abstractions.append({
                "name": abs_.name,
                "systems": sorted(core_systems),
                "cycle": abs_.invention_cycle,
                "reuse": abs_.reuse_count,
            })
            depths.append(len(core_systems))

            # Count pairs
            systems_list = sorted(core_systems)
            for i in range(len(systems_list)):
                for j in range(i + 1, len(systems_list)):
                    pair = (systems_list[i], systems_list[j])
                    pair_counts[pair] += 1

    report.cross_system_rate = (
        report.cross_system_count / report.total_abstractions
        if report.total_abstractions > 0 else 0.0
    )
    report.system_pair_counts = dict(pair_counts)
    report.integration_depth = sum(depths) / len(depths) if depths else 0.0

    # Growth by cycle
    by_cycle = library.abstractions_by_cycle()
    for cycle in sorted(by_cycle.keys()):
        cycle_abs = by_cycle[cycle]
        cross = sum(1 for a in cycle_abs if a.is_cross_system)
        report.growth_by_cycle.append({
            "cycle": cycle,
            "new_abstractions": len(cycle_abs),
            "new_cross_system": cross,
            "cumulative_cross_system": sum(
                1 for a in library.abstractions
                if a.invention_cycle <= cycle and a.is_cross_system
            ),
        })

    return report


def detect_placeholder_patterns(library: Library) -> list[dict]:
    """
    Detect Carey-style placeholder patterns in library growth.

    A placeholder pattern is when:
    1. An abstraction is first used with partial content (one system)
    2. Later, a more complex abstraction uses it with additional systems

    This mirrors the developmental sequence: count-list → subset-knower → cardinal principle.
    """
    patterns = []

    # Sort abstractions by cycle
    by_cycle = library.abstractions_by_cycle()

    for cycle in sorted(by_cycle.keys()):
        for abs_ in by_cycle[cycle]:
            # Check if this abstraction extends an earlier one
            for earlier_abs in library.abstractions:
                if earlier_abs.invention_cycle >= cycle:
                    continue
                if earlier_abs.name in _get_used_abstractions(abs_):
                    # This abstraction extends an earlier one
                    new_systems = abs_.systems_composed - earlier_abs.systems_composed
                    if new_systems & CORE_SYSTEMS:
                        patterns.append({
                            "placeholder": earlier_abs.name,
                            "placeholder_systems": sorted(earlier_abs.systems_composed),
                            "extended_by": abs_.name,
                            "new_systems": sorted(new_systems),
                            "cycle_gap": cycle - earlier_abs.invention_cycle,
                        })

    return patterns


def _get_used_abstractions(abs_: Abstraction) -> set[str]:
    """Get names of library abstractions used in this abstraction's body."""
    if abs_.body is None:
        return set()
    prims = abs_.body.primitives_used()
    # Filter to only invented abstractions (not base primitives)
    return {p for p in prims if p.startswith("abs_")}
