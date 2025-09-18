"""
analogy.py — Gentner-Style Structural Alignment Primitive

Implements the cross-domain analogy operator based on Gentner's
Structure-Mapping Theory (1983) and the Structure-Mapping Engine
(Falkenhainer, Forbus & Gentner 1989).

This is the mechanism by which representations from distinct Spelke
core systems get integrated — the engine of Carey-style bootstrapping.

Key principle: analogy = alignment of relational structure between base
and target, with systematicity preference (higher-order interconnected
relations preferred over isolated attribute matches).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tint, tbool, tlist, tpair, tmapping,
)


# ──────────────────────────────────────────────────────────────────────
# Structural Representation (Object Graphs)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ObjectNode:
    """A node in the object graph — represents one entity."""
    id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    # Attributes: color, size, shape_hash, position, etc.

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, ObjectNode) and self.id == other.id


@dataclass
class Relation:
    """A relation between nodes in the object graph."""
    name: str
    args: tuple[str, ...]  # IDs of related nodes
    order: int = 1  # 1 = first-order, 2 = higher-order

    def __hash__(self):
        return hash((self.name, self.args, self.order))


@dataclass
class ObjectGraph:
    """Graph representation of a grid scene — nodes + relations."""
    nodes: dict[str, ObjectNode] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    def add_node(self, node: ObjectNode):
        self.nodes[node.id] = node

    def add_relation(self, rel: Relation):
        self.relations.append(rel)

    @property
    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def relations_involving(self, node_id: str) -> list[Relation]:
        return [r for r in self.relations if node_id in r.args]


@dataclass
class StructuralMapping:
    """Result of structural alignment between two object graphs."""
    node_mapping: dict[str, str]  # base_id -> target_id
    relation_mapping: dict[int, int]  # index in base -> index in target
    score: float = 0.0
    candidate_inferences: list[Relation] = field(default_factory=list)

    def apply_to_node(self, base_node_id: str) -> Optional[str]:
        return self.node_mapping.get(base_node_id)

    def unmapped_base_relations(self, base_graph: ObjectGraph) -> list[Relation]:
        mapped_indices = set(self.relation_mapping.keys())
        return [r for i, r in enumerate(base_graph.relations) if i not in mapped_indices]


# ──────────────────────────────────────────────────────────────────────
# Grid → Object Graph conversion
# ──────────────────────────────────────────────────────────────────────

def grid_to_object_graph(grid: np.ndarray, objects: list = None) -> ObjectGraph:
    """
    Convert a grid (with extracted objects) into a structural object graph.
    This is the bridge from perceptual representation to relational representation.
    """
    from src.spelke_dsl.l_objects import _extract_objects, GridObject

    if objects is None:
        objects = _extract_objects(grid)

    graph = ObjectGraph()

    # Create nodes
    for i, obj in enumerate(objects):
        node = ObjectNode(
            id=f"obj_{i}",
            attributes={
                "color": obj.color,
                "size": obj.size,
                "height": obj.height,
                "width": obj.width,
                "row": obj.top_left[0],
                "col": obj.top_left[1],
                "center_r": obj.center[0],
                "center_c": obj.center[1],
                "shape_hash": hash(obj.cells),
            }
        )
        graph.add_node(node)

    # Extract spatial relations
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            oi, oj = objects[i], objects[j]
            ni, nj = f"obj_{i}", f"obj_{j}"

            # Relative position
            if oi.center[0] < oj.center[0]:
                graph.add_relation(Relation("above", (ni, nj)))
            elif oi.center[0] > oj.center[0]:
                graph.add_relation(Relation("below", (ni, nj)))

            if oi.center[1] < oj.center[1]:
                graph.add_relation(Relation("left_of", (ni, nj)))
            elif oi.center[1] > oj.center[1]:
                graph.add_relation(Relation("right_of", (ni, nj)))

            # Same properties
            if oi.color == oj.color:
                graph.add_relation(Relation("same_color", (ni, nj)))
            if oi.size == oj.size:
                graph.add_relation(Relation("same_size", (ni, nj)))
            if oi.mask.shape == oj.mask.shape and np.array_equal(oi.mask, oj.mask):
                graph.add_relation(Relation("same_shape", (ni, nj)))

            # Size relations
            if oi.size > oj.size:
                graph.add_relation(Relation("larger_than", (ni, nj)))
            elif oi.size < oj.size:
                graph.add_relation(Relation("smaller_than", (ni, nj)))

            # Adjacency
            from src.spelke_dsl.l_objects import _touching
            if _touching(oi, oj):
                graph.add_relation(Relation("touching", (ni, nj)))

    # Higher-order relations (Gentner's systematicity preference)
    # Group detection
    color_groups: dict[int, list[str]] = {}
    for i, obj in enumerate(objects):
        color_groups.setdefault(obj.color, []).append(f"obj_{i}")
    for color, members in color_groups.items():
        if len(members) > 1:
            graph.add_relation(Relation("color_group", tuple(members), order=2))

    return graph


# ──────────────────────────────────────────────────────────────────────
# Structure-Mapping Engine (SME) — Core Algorithm
# ──────────────────────────────────────────────────────────────────────

def _compute_attribute_similarity(n1: ObjectNode, n2: ObjectNode) -> float:
    """Compute attribute-level similarity between two nodes."""
    if not n1.attributes or not n2.attributes:
        return 0.0

    shared_keys = set(n1.attributes.keys()) & set(n2.attributes.keys())
    if not shared_keys:
        return 0.0

    matches = 0
    for key in shared_keys:
        v1, v2 = n1.attributes[key], n2.attributes[key]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # Ratio similarity for numeric
            if v1 == 0 and v2 == 0:
                matches += 1.0
            elif v1 == 0 or v2 == 0:
                continue
            else:
                ratio = min(v1, v2) / max(v1, v2)
                matches += ratio
        elif v1 == v2:
            matches += 1.0

    return matches / len(shared_keys)


def _compute_relation_match(r1: Relation, r2: Relation) -> float:
    """Score how well two relations match (name + arity + order)."""
    if r1.name != r2.name:
        return 0.0
    if len(r1.args) != len(r2.args):
        return 0.0
    # Higher-order relations get a systematicity bonus
    base_score = 1.0
    if r1.order > 1:
        base_score = 2.0  # Gentner's systematicity preference
    return base_score


def structural_alignment(
    base: ObjectGraph,
    target: ObjectGraph,
    max_mappings: int = 100,
) -> list[StructuralMapping]:
    """
    Gentner-style structural alignment between two object graphs.

    Algorithm (simplified SME):
    1. Find local matches: compatible node pairs + relation pairs
    2. Build mapping candidates via greedy consistent extension
    3. Score by structural consistency (relation preservation > attribute match)
    4. Return top mappings ranked by score

    Systematicity preference: higher-order relations weighted more.
    """
    base_nodes = list(base.nodes.values())
    target_nodes = list(target.nodes.values())

    if not base_nodes or not target_nodes:
        return [StructuralMapping({}, {}, 0.0)]

    # Step 1: Compute pairwise node compatibility
    compatibility = {}
    for bn in base_nodes:
        for tn in target_nodes:
            sim = _compute_attribute_similarity(bn, tn)
            if sim > 0.0:
                compatibility[(bn.id, tn.id)] = sim

    # Step 2: Greedy mapping construction
    # Start with highest-compatibility pairs, extend consistently
    sorted_pairs = sorted(compatibility.items(), key=lambda x: -x[1])

    mappings = []

    for seed_pair, seed_score in sorted_pairs[:max_mappings]:
        base_id, target_id = seed_pair
        mapping = {base_id: target_id}
        used_targets = {target_id}
        total_score = seed_score

        # Extend: for each relation in base involving mapped nodes,
        # find matching relation in target and extend mapping
        for bi, brel in enumerate(base.relations):
            # Check if any args are already mapped
            mapped_args = [a for a in brel.args if a in mapping]
            if not mapped_args:
                continue

            # Find matching target relation
            for ti, trel in enumerate(target.relations):
                rmatch = _compute_relation_match(brel, trel)
                if rmatch == 0:
                    continue

                # Check consistency with existing mapping
                consistent = True
                new_mappings = {}
                for ba, ta in zip(brel.args, trel.args):
                    if ba in mapping:
                        if mapping[ba] != ta:
                            consistent = False
                            break
                    elif ta in used_targets:
                        consistent = False
                        break
                    else:
                        new_mappings[ba] = ta

                if consistent:
                    mapping.update(new_mappings)
                    used_targets.update(new_mappings.values())
                    total_score += rmatch
                    break

        # Candidate inferences: relations in base not yet matched
        sm = StructuralMapping(mapping, {}, total_score)
        sm.candidate_inferences = sm.unmapped_base_relations(base)
        mappings.append(sm)

    # Deduplicate and sort
    seen = set()
    unique_mappings = []
    for m in mappings:
        key = frozenset(m.node_mapping.items())
        if key not in seen:
            seen.add(key)
            unique_mappings.append(m)

    unique_mappings.sort(key=lambda m: -m.score)
    return unique_mappings if unique_mappings else [StructuralMapping({}, {}, 0.0)]


# ──────────────────────────────────────────────────────────────────────
# Analogical transfer
# ──────────────────────────────────────────────────────────────────────

def _analogical_transfer(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    new_input: np.ndarray,
) -> np.ndarray:
    """
    The full analogy primitive:
    1. Build object graphs for input, output, and new_input
    2. Align input→output to discover the transformation
    3. Align input→new_input to find correspondences
    4. Apply the discovered transformation to new_input

    This is the cross-domain transfer mechanism.
    """
    from src.spelke_dsl.l_objects import _extract_objects

    # Build graphs
    input_objs = _extract_objects(input_grid)
    output_objs = _extract_objects(output_grid)
    new_objs = _extract_objects(new_input)

    g_in = grid_to_object_graph(input_grid, input_objs)
    g_out = grid_to_object_graph(output_grid, output_objs)
    g_new = grid_to_object_graph(new_input, new_objs)

    # Align input→output (discover transformation)
    transform_mappings = structural_alignment(g_in, g_out)
    if not transform_mappings:
        return new_input.copy()

    best_transform = transform_mappings[0]

    # Align input→new_input (find correspondences)
    correspond_mappings = structural_alignment(g_in, g_new)
    if not correspond_mappings:
        return new_input.copy()

    best_correspond = correspond_mappings[0]

    # Apply transformation: for each mapped base→target pair in transform,
    # find the corresponding new_input object and apply the change
    result = new_input.copy()

    for base_id, out_id in best_transform.node_mapping.items():
        # Find what this base node maps to in new_input
        new_id = best_correspond.node_mapping.get(base_id)
        if new_id is None:
            continue

        # Get the objects
        base_idx = int(base_id.split("_")[1]) if "_" in base_id else 0
        out_idx = int(out_id.split("_")[1]) if "_" in out_id else 0
        new_idx = int(new_id.split("_")[1]) if "_" in new_id else 0

        if base_idx >= len(input_objs) or out_idx >= len(output_objs) or new_idx >= len(new_objs):
            continue

        in_obj = input_objs[base_idx]
        out_obj = output_objs[out_idx]
        new_obj = new_objs[new_idx]

        # Detect transformation type and apply
        # Color change
        if in_obj.color != out_obj.color:
            for r, c in new_obj.cells:
                if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                    result[r, c] = out_obj.color

        # Position change (translation)
        dr = out_obj.top_left[0] - in_obj.top_left[0]
        dc = out_obj.top_left[1] - in_obj.top_left[1]
        if dr != 0 or dc != 0:
            # Clear old position
            for r, c in new_obj.cells:
                if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                    result[r, c] = 0
            # Place at new position
            color = out_obj.color if in_obj.color != out_obj.color else new_obj.color
            for r, c in new_obj.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                    result[nr, nc] = color

    return result


def _align_grids(g1: np.ndarray, g2: np.ndarray) -> list[StructuralMapping]:
    """Structural alignment between two grids."""
    graph1 = grid_to_object_graph(g1)
    graph2 = grid_to_object_graph(g2)
    return structural_alignment(graph1, graph2)


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_analogy_primitives(registry: PrimitiveRegistry) -> None:
    """Register the Gentner-style analogy primitives."""
    S = SpelkeSystem.ANALOGY

    primitives = [
        Primitive(
            "analogy_transfer",
            Arrow(tgrid, Arrow(tgrid, Arrow(tgrid, tgrid))),
            lambda inp: lambda out: lambda new: _analogical_transfer(inp, out, new),
            S,
            "Gentner-style analogical transfer: learn transformation from "
            "input→output, apply to new_input",
        ),
        Primitive(
            "align_grids",
            Arrow(tgrid, Arrow(tgrid, tmapping)),
            lambda g1: lambda g2: _align_grids(g1, g2),
            S,
            "Structural alignment between two grids",
        ),
        Primitive(
            "grid_to_graph",
            Arrow(tgrid, tmapping),
            lambda g: grid_to_object_graph(g),
            S,
            "Convert grid to object graph representation",
        ),
    ]

    for p in primitives:
        registry.register(p)
