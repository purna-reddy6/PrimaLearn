"""
l_persons.py — PERSONS Core Knowledge Module (Spelke System 3)

Implements Spelke's PERSONS system: face/agent preference, intentional
action attribution, social distance, and theory-of-mind proxies.

Developmental evidence (Johnson & Morton 1991, Gergely 2002, Csibra &
Gergely 2009): infants distinguish persons from objects via asymmetry,
directionality, and goal-directed action patterns.

In the ARC domain, "persons" manifest as:
  - Asymmetric/elongated objects (agents) vs compact/symmetric objects (targets)
  - Directional facing based on longest axis
  - Social distance between agent-like objects
  - Goal-reachability (clear path to target)
  - Mirror intent (reflection as social mirroring)

Design Note: PERSONS primitives encode the social-cognition signatures of
agent detection and intentional-action parsing. They complement AGENTS
(goal-directed movement) and OBJECTS (physical cohesion).
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tpoint, tlist, tpair,
)


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _extract_objects_list(grid: np.ndarray) -> list[np.ndarray]:
    """Return list of bounding-box sub-grids, one per connected component."""
    from collections import deque

    if grid.size == 0:
        return []

    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for start_r in range(h):
        for start_c in range(w):
            val = int(grid[start_r, start_c])
            if val == 0 or visited[start_r, start_c]:
                continue
            # BFS to collect connected component
            cells = []
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == val:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if cells:
                min_r = min(r for r, _ in cells)
                max_r = max(r for r, _ in cells)
                min_c = min(c for _, c in cells)
                max_c = max(c for _, c in cells)
                obj = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
                for r, c in cells:
                    obj[r - min_r, c - min_c] = int(grid[r, c])
                objects.append(obj)

    return objects


def _aspect_ratio(obj: np.ndarray) -> float:
    """Return max(H/W, W/H) — elongation measure."""
    if obj.size == 0:
        return 1.0
    h, w = obj.shape
    if w == 0 or h == 0:
        return 1.0
    return max(h / w, w / h) if min(h, w) > 0 else 1.0


def _centroid(obj: np.ndarray) -> tuple[float, float]:
    """Center of mass of non-zero cells."""
    positions = np.argwhere(obj != 0)
    if len(positions) == 0:
        return (0.0, 0.0)
    return (float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1])))


def _obj_color(obj: np.ndarray) -> int:
    """Dominant (most frequent) non-zero color in object."""
    vals = obj[obj != 0]
    if len(vals) == 0:
        return 0
    counts = np.bincount(vals)
    counts[0] = 0
    return int(np.argmax(counts))


# ──────────────────────────────────────────────────────────────────────
# Detection Primitives
# ──────────────────────────────────────────────────────────────────────

def _is_agent_obj(obj_grid: np.ndarray) -> bool:
    """True if object is elongated/asymmetric (agent-like). Aspect ratio > 1.5."""
    if obj_grid.size == 0:
        return False
    return _aspect_ratio(obj_grid) > 1.5


def _is_target_obj(obj_grid: np.ndarray) -> bool:
    """True if object is compact/symmetric (target-like). Aspect ratio < 1.3, area > 2."""
    if obj_grid.size == 0:
        return False
    area = int(np.count_nonzero(obj_grid))
    return _aspect_ratio(obj_grid) < 1.3 and area > 2


def _agent_count(grid: np.ndarray) -> int:
    """Count asymmetric (agent-like) objects in grid."""
    if grid.size == 0:
        return 0
    objects = _extract_objects_list(grid)
    return sum(1 for obj in objects if _is_agent_obj(obj))


# ──────────────────────────────────────────────────────────────────────
# Trajectory / Intentionality Primitives
# ──────────────────────────────────────────────────────────────────────

def _obj_facing_dir(obj_grid: np.ndarray) -> int:
    """
    Facing direction based on longest bounding-box axis.
    Returns 0=up (tall), 1=right (wide), 2=down (tall, alt), 3=left (wide, alt).
    Simplified: 0=up if taller, 1=right if wider.
    """
    if obj_grid.size == 0:
        return 0
    h, w = obj_grid.shape
    if h >= w:
        return 0  # tall → facing up
    else:
        return 1  # wide → facing right


def _point_toward(obj: np.ndarray, target: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Rotate obj to face target's position in grid.
    Returns grid with obj oriented toward target (transpose if needed).
    """
    result = grid.copy()
    if grid.size == 0 or obj.size == 0 or target.size == 0:
        return result

    obj_positions = []
    target_positions = []
    obj_color = _obj_color(obj)
    target_color = _obj_color(target)

    if obj_color == 0 or target_color == 0:
        return result

    obj_pos_arr = np.argwhere(grid == obj_color)
    tgt_pos_arr = np.argwhere(grid == target_color)

    if len(obj_pos_arr) == 0 or len(tgt_pos_arr) == 0:
        return result

    # Centroid of agent and target in the full grid
    agent_r = float(np.mean(obj_pos_arr[:, 0]))
    agent_c = float(np.mean(obj_pos_arr[:, 1]))
    target_r = float(np.mean(tgt_pos_arr[:, 0]))
    target_c = float(np.mean(tgt_pos_arr[:, 1]))

    dr = target_r - agent_r
    dc = target_c - agent_c

    # Use agent's bounding box (not full grid shape) for orientation check
    min_r = int(np.min(obj_pos_arr[:, 0]))
    max_r = int(np.max(obj_pos_arr[:, 0]))
    min_c = int(np.min(obj_pos_arr[:, 1]))
    max_c = int(np.max(obj_pos_arr[:, 1]))
    agent_h = max_r - min_r + 1
    agent_w = max_c - min_c + 1
    is_tall = agent_h >= agent_w
    should_be_tall = abs(dr) >= abs(dc)  # target is more vertical → face up/down

    if is_tall == should_be_tall:
        return result  # Already oriented correctly

    # Rotate: transpose agent cells in-place
    result2 = result.copy()

    # Clear old agent cells
    for r, c in obj_pos_arr:
        result2[r, c] = 0

    # Transposed dimensions
    new_h, new_w = agent_w, agent_h
    center_r = int(agent_r)
    center_c = int(agent_c)
    start_r = center_r - new_h // 2
    start_c = center_c - new_w // 2
    grid_h, grid_w = grid.shape

    # Source bounding box: agent at (min_r..max_r, min_c..max_c) in grid
    for r in range(agent_h):
        for c in range(agent_w):
            src_val = int(grid[min_r + r, min_c + c])
            if src_val == obj_color:
                # Transpose: (r, c) → (c, r) in new coordinate space
                new_r = start_r + c
                new_c = start_c + r
                if 0 <= new_r < grid_h and 0 <= new_c < grid_w:
                    result2[new_r, new_c] = src_val

    return result2


def _social_dist(obj1: np.ndarray, obj2: np.ndarray, grid: np.ndarray) -> int:
    """Manhattan distance between centroids of two objects in the grid (by color)."""
    if obj1.size == 0 or obj2.size == 0 or grid.size == 0:
        return 0

    c1 = _obj_color(obj1)
    c2 = _obj_color(obj2)
    if c1 == 0 or c2 == 0:
        return 0

    pos1 = np.argwhere(grid == c1)
    pos2 = np.argwhere(grid == c2)

    if len(pos1) == 0 or len(pos2) == 0:
        return 0

    r1, c1_coord = float(np.mean(pos1[:, 0])), float(np.mean(pos1[:, 1]))
    r2, c2_coord = float(np.mean(pos2[:, 0])), float(np.mean(pos2[:, 1]))

    return int(abs(r1 - r2) + abs(c1_coord - c2_coord))


# ──────────────────────────────────────────────────────────────────────
# Social Composition Primitives
# ──────────────────────────────────────────────────────────────────────

def _extract_agents(grid: np.ndarray) -> np.ndarray:
    """Mask of all agent-like (elongated) objects."""
    result = np.zeros_like(grid)
    if grid.size == 0:
        return result
    objects = _extract_objects_list(grid)
    for obj in objects:
        if _is_agent_obj(obj):
            c = _obj_color(obj)
            if c != 0:
                result[grid == c] = c
    return result


def _extract_targets(grid: np.ndarray) -> np.ndarray:
    """Mask of all target-like (compact) objects."""
    result = np.zeros_like(grid)
    if grid.size == 0:
        return result
    objects = _extract_objects_list(grid)
    for obj in objects:
        if _is_target_obj(obj):
            c = _obj_color(obj)
            if c != 0:
                result[grid == c] = c
    return result


def _nearest_agent(grid: np.ndarray) -> np.ndarray:
    """Return grid containing only the largest asymmetric (agent-like) object."""
    result = np.zeros_like(grid)
    if grid.size == 0:
        return result
    objects = _extract_objects_list(grid)
    agent_objs = [obj for obj in objects if _is_agent_obj(obj)]
    if not agent_objs:
        return result
    # Largest by area
    largest = max(agent_objs, key=lambda o: int(np.count_nonzero(o)))
    c = _obj_color(largest)
    if c != 0:
        result[grid == c] = c
    return result


def _nearest_target(grid: np.ndarray, agent: np.ndarray) -> np.ndarray:
    """Return grid containing only the target-like object closest to agent."""
    result = np.zeros_like(grid)
    if grid.size == 0 or agent.size == 0:
        return result

    agent_color = _obj_color(agent)
    if agent_color == 0:
        # Try to find any agent-like object
        agent_mask = _nearest_agent(grid)
        agent_color = _obj_color(agent_mask)

    if agent_color == 0:
        return result

    # Agent centroid in grid
    agent_positions = np.argwhere(grid == agent_color)
    if len(agent_positions) == 0:
        return result

    agent_r = float(np.mean(agent_positions[:, 0]))
    agent_c = float(np.mean(agent_positions[:, 1]))

    objects = _extract_objects_list(grid)
    target_objs = [obj for obj in objects if _is_target_obj(obj)]
    if not target_objs:
        return result

    # Find closest target by centroid distance
    best_obj = None
    best_dist = float('inf')
    for obj in target_objs:
        c = _obj_color(obj)
        if c == 0 or c == agent_color:
            continue
        pos = np.argwhere(grid == c)
        if len(pos) == 0:
            continue
        tr = float(np.mean(pos[:, 0]))
        tc = float(np.mean(pos[:, 1]))
        dist = abs(tr - agent_r) + abs(tc - agent_c)
        if dist < best_dist:
            best_dist = dist
            best_obj = obj

    if best_obj is not None:
        c = _obj_color(best_obj)
        if c != 0:
            result[grid == c] = c

    return result


# ──────────────────────────────────────────────────────────────────────
# Theory of Mind (Simple ARC proxies)
# ──────────────────────────────────────────────────────────────────────

def _goal_reachable(agent: np.ndarray, target: np.ndarray, grid: np.ndarray) -> bool:
    """
    True if there is a clear (BFS) path from agent to target in the grid.
    Obstacles are non-zero cells that are neither agent nor target color.
    """
    if agent.size == 0 or target.size == 0 or grid.size == 0:
        return False

    agent_color = _obj_color(agent)
    target_color = _obj_color(target)
    if agent_color == 0 or target_color == 0:
        return False

    agent_positions = np.argwhere(grid == agent_color)
    target_positions = np.argwhere(grid == target_color)
    if len(agent_positions) == 0 or len(target_positions) == 0:
        return False

    from collections import deque
    h, w = grid.shape

    # Start from agent centroid cell
    start = (int(np.mean(agent_positions[:, 0])), int(np.mean(agent_positions[:, 1])))
    # Find nearest actual agent cell to centroid
    dists = [abs(r - start[0]) + abs(c - start[1]) for r, c in agent_positions]
    start = tuple(agent_positions[int(np.argmin(dists))].tolist())

    target_set = {tuple(p.tolist()) for p in target_positions}

    visited = {start}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) in target_set:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                cell = int(grid[nr, nc])
                if cell == 0 or cell == agent_color or cell == target_color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def _mirror_intent(grid: np.ndarray) -> np.ndarray:
    """
    Reflect grid based on facing direction of the largest agent-like object.
    If agent faces right (wide), flip horizontally. If up (tall), flip vertically.
    """
    if grid.size == 0:
        return grid.copy()

    agent_mask = _nearest_agent(grid)
    agent_color = _obj_color(agent_mask)

    if agent_color == 0:
        return np.fliplr(grid.copy())

    agent_positions = np.argwhere(grid == agent_color)
    if len(agent_positions) == 0:
        return np.fliplr(grid.copy())

    # Bounding box of agent
    min_r, max_r = int(np.min(agent_positions[:, 0])), int(np.max(agent_positions[:, 0]))
    min_c, max_c = int(np.min(agent_positions[:, 1])), int(np.max(agent_positions[:, 1]))
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Tall → flip vertically (facing up/down)
    # Wide → flip horizontally (facing left/right)
    if h >= w:
        return np.flipud(grid.copy())
    else:
        return np.fliplr(grid.copy())


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_persons_primitives(registry: PrimitiveRegistry) -> None:
    """Register all PERSONS core knowledge primitives."""
    S = SpelkeSystem.PERSONS

    primitives = [
        # ── Detection ──
        Primitive(
            "is_agent_obj", Arrow(tgrid, tbool),
            lambda g: _is_agent_obj(g),
            S, "True if object is elongated/asymmetric (agent-like, aspect ratio > 1.5)",
            spelke_properties=["agent-detection", "asymmetry"],
        ),
        Primitive(
            "is_target_obj", Arrow(tgrid, tbool),
            lambda g: _is_target_obj(g),
            S, "True if object is compact/symmetric (target-like, aspect ratio < 1.3, area > 2)",
            spelke_properties=["agent-detection", "symmetry"],
        ),
        Primitive(
            "agent_count", Arrow(tgrid, tint),
            lambda g: _agent_count(g),
            S, "Count asymmetric (agent-like) objects in grid",
            spelke_properties=["agent-detection", "numerosity"],
        ),

        # ── Trajectory / Intentionality ──
        Primitive(
            "obj_facing_dir", Arrow(tgrid, tint),
            lambda g: _obj_facing_dir(g),
            S, "Facing direction: 0=up (tall), 1=right (wide)",
            spelke_properties=["intentionality", "directionality"],
        ),
        Primitive(
            "point_toward",
            Arrow(tgrid, Arrow(tgrid, Arrow(tgrid, tgrid))),
            lambda obj: lambda tgt: lambda g: _point_toward(obj, tgt, g),
            S, "Rotate obj to face target's position in grid",
            spelke_properties=["intentionality", "goal-orientation"],
        ),
        Primitive(
            "social_dist",
            Arrow(tgrid, Arrow(tgrid, Arrow(tgrid, tint))),
            lambda o1: lambda o2: lambda g: _social_dist(o1, o2, g),
            S, "Manhattan distance between centroids of two objects in grid",
            spelke_properties=["social-distance", "proximity"],
        ),

        # ── Social composition ──
        Primitive(
            "extract_agents", Arrow(tgrid, tgrid),
            lambda g: _extract_agents(g),
            S, "Mask of all agent-like (elongated) objects in grid",
            spelke_properties=["agent-detection"],
        ),
        Primitive(
            "extract_targets", Arrow(tgrid, tgrid),
            lambda g: _extract_targets(g),
            S, "Mask of all target-like (compact) objects in grid",
            spelke_properties=["agent-detection"],
        ),
        Primitive(
            "nearest_agent", Arrow(tgrid, tgrid),
            lambda g: _nearest_agent(g),
            S, "Grid containing only the largest asymmetric (agent-like) object",
            spelke_properties=["agent-detection", "social-salience"],
        ),
        Primitive(
            "nearest_target",
            Arrow(tgrid, Arrow(tgrid, tgrid)),
            lambda g: lambda agent: _nearest_target(g, agent),
            S, "Grid containing the target-like object closest to agent",
            spelke_properties=["goal-inference", "social-salience"],
        ),

        # ── Theory of mind ──
        Primitive(
            "goal_reachable",
            Arrow(tgrid, Arrow(tgrid, Arrow(tgrid, tbool))),
            lambda agent: lambda target: lambda g: _goal_reachable(agent, target, g),
            S, "True if there is a clear path from agent to target in grid",
            spelke_properties=["theory-of-mind", "reachability"],
        ),
        Primitive(
            "mirror_intent", Arrow(tgrid, tgrid),
            lambda g: _mirror_intent(g),
            S, "Flip grid based on agent facing direction (social mirroring)",
            spelke_properties=["theory-of-mind", "mirror-neurons"],
        ),
    ]

    for p in primitives:
        registry.register(p)


"""
PERSONS module primitives: 12 total
- 3 detection (is_agent_obj, is_target_obj, agent_count)
- 4 trajectory/intentionality (obj_facing_dir, point_toward, social_dist, nearest_agent)
  [nearest_agent placed here as it returns the primary agent, used for trajectory]
- 3 social composition (extract_agents, extract_targets, nearest_target)
- 2 theory of mind (goal_reachable, mirror_intent)
"""
