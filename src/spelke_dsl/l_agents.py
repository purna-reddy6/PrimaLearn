"""
l_agents.py — AGENTS Core Knowledge Module (Spelke System 2)

Implements Spelke's AGENTS system: goal-directedness, equifinality,
rational efficiency, and basic action-outcome contingency.

Developmental evidence (Woodward 1998, Gergely et al. 2002, Csibra &
Gergely 2009): infants as young as 5 months attribute goals to agents
based on the efficiency of their actions toward an end-state.

In the ARC domain, "agents" manifest as:
  - Directed movement toward a target (pathfinding)
  - Rational efficiency (shortest path preference)
  - Goal inference from trajectory
  - Contingency detection (action→outcome pairing)
  - Object transport (agent moves object toward goal)

Design Note: These primitives encode the COMPUTATIONAL SIGNATURES of
goal-directed behavior, not full planning. They complement OBJECTS
(what moves) and PLACES (where things are) to enable cross-system
abstractions like "agent-moved-object-toward-goal."
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tpoint, tlist, tpair,
)


# ──────────────────────────────────────────────────────────────────────
# Goal-Directed Movement Primitives
# ──────────────────────────────────────────────────────────────────────

def _find_colored_cell(grid: np.ndarray, color: int) -> Optional[tuple]:
    """Find the first cell of a given color (row, col)."""
    positions = np.argwhere(grid == color)
    if len(positions) == 0:
        return None
    return (int(positions[0, 0]), int(positions[0, 1]))


def _find_all_colored_cells(grid: np.ndarray, color: int) -> list[tuple]:
    """Find all cells of a given color."""
    positions = np.argwhere(grid == color)
    return [(int(r), int(c)) for r, c in positions]


def _manhattan_distance(p1: tuple, p2: tuple) -> int:
    """Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _trace_path(grid: np.ndarray, agent_color: int, goal_color: int) -> list[tuple]:
    """
    Trace the shortest path from agent to goal on the grid.
    BFS pathfinding treating non-background, non-agent, non-goal cells as obstacles.
    Returns path as list of (row, col) coordinates.
    """
    from collections import deque

    start = _find_colored_cell(grid, agent_color)
    end = _find_colored_cell(grid, goal_color)
    if start is None or end is None:
        return []

    h, w = grid.shape
    visited = set()
    queue = deque([(start, [start])])
    visited.add(start)

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                cell_val = int(grid[nr, nc])
                # Can traverse: background (0), goal color, or agent color
                if cell_val == 0 or cell_val == goal_color or cell_val == agent_color:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))

    return []  # No path found


def _draw_path(
    grid: np.ndarray, agent_color: int, goal_color: int, path_color: int = -1
) -> np.ndarray:
    """
    Draw the shortest path from agent to goal on the grid.
    If path_color is -1, use the agent's color for the path.
    """
    result = grid.copy()
    path = _trace_path(grid, agent_color, goal_color)
    if not path:
        return result

    if path_color == -1:
        path_color = agent_color

    for r, c in path[1:-1]:  # Exclude start and end
        result[r, c] = path_color

    return result


def _move_toward(
    grid: np.ndarray, agent_color: int, goal_color: int
) -> np.ndarray:
    """
    Move the agent one step toward the goal (rational efficiency).
    The agent moves along the shortest path direction.
    """
    result = grid.copy()
    start = _find_colored_cell(grid, agent_color)
    end = _find_colored_cell(grid, goal_color)
    if start is None or end is None:
        return result

    path = _trace_path(grid, agent_color, goal_color)
    if len(path) < 2:
        return result

    # Move agent to the next step
    next_pos = path[1]
    # Clear old position
    result[start[0], start[1]] = 0
    # Place at new position (unless it's the goal)
    if next_pos != end:
        result[next_pos[0], next_pos[1]] = agent_color

    return result


def _move_to_goal(
    grid: np.ndarray, agent_color: int, goal_color: int
) -> np.ndarray:
    """
    Move the agent all the way to the goal position.
    Clear the agent's original position, place it at the goal.
    """
    result = grid.copy()
    agent_cells = _find_all_colored_cells(grid, agent_color)
    goal_pos = _find_colored_cell(grid, goal_color)
    if not agent_cells or goal_pos is None:
        return result

    # Clear agent
    for r, c in agent_cells:
        result[r, c] = 0

    # Place at goal
    result[goal_pos[0], goal_pos[1]] = agent_color
    return result


# ──────────────────────────────────────────────────────────────────────
# Goal Inference Primitives
# ──────────────────────────────────────────────────────────────────────

def _infer_goal_direction(grid: np.ndarray, agent_color: int) -> int:
    """
    Infer the likely goal direction from agent position.
    Returns direction encoding: 0=up, 1=down, 2=left, 3=right.

    Heuristic: the goal is in the direction with the most non-background
    cells (most structure).
    """
    pos = _find_colored_cell(grid, agent_color)
    if pos is None:
        return 0

    h, w = grid.shape
    r, c = pos

    # Count non-background cells in each direction
    up_count = np.count_nonzero(grid[:r, :]) if r > 0 else 0
    down_count = np.count_nonzero(grid[r+1:, :]) if r < h - 1 else 0
    left_count = np.count_nonzero(grid[:, :c]) if c > 0 else 0
    right_count = np.count_nonzero(grid[:, c+1:]) if c < w - 1 else 0

    counts = [up_count, down_count, left_count, right_count]
    return int(np.argmax(counts))


def _is_rational_path(grid: np.ndarray, path_color: int,
                       agent_color: int, goal_color: int) -> bool:
    """
    Evaluate whether a colored path on the grid is the shortest path
    between agent and goal (rational efficiency criterion).

    Infants expect agents to take the most efficient path to their goal
    (Gergely & Csibra 2003).
    """
    start = _find_colored_cell(grid, agent_color)
    end = _find_colored_cell(grid, goal_color)
    if start is None or end is None:
        return False

    # Find shortest path length
    shortest = _trace_path(grid, agent_color, goal_color)
    if not shortest:
        return False

    # Count path cells on the grid
    actual_path_cells = _find_all_colored_cells(grid, path_color)
    # Rational if actual path length ≤ shortest path length + 1
    return len(actual_path_cells) <= len(shortest) + 1


# ──────────────────────────────────────────────────────────────────────
# Contingency Detection
# ──────────────────────────────────────────────────────────────────────

def _detect_contingency(
    input_grid: np.ndarray, output_grid: np.ndarray
) -> list[tuple]:
    """
    Detect action→outcome contingencies between input and output grids.
    Returns list of (source_color, outcome_description) pairs.

    This is the fundamental agent primitive: detecting that an action
    (change in input) caused an outcome (change in output).
    """
    if input_grid.shape != output_grid.shape:
        return []

    contingencies = []
    diff = input_grid != output_grid
    if not diff.any():
        return []

    # Find which colors changed
    changed_from = set(input_grid[diff].tolist())
    changed_to = set(output_grid[diff].tolist())

    for src in changed_from:
        for dst in changed_to:
            if src != dst:
                contingencies.append((int(src), int(dst)))

    return contingencies


def _project_trajectory(
    grid: np.ndarray, agent_color: int, direction: int, steps: int = -1
) -> np.ndarray:
    """
    Project the agent's trajectory in a given direction.
    direction: 0=up, 1=down, 2=left, 3=right.
    steps: number of steps (-1 = until hitting obstacle or edge).

    Returns grid with trajectory marked.
    """
    result = grid.copy()
    pos = _find_colored_cell(grid, agent_color)
    if pos is None:
        return result

    dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction]
    r, c = pos
    h, w = grid.shape
    step_count = 0

    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if grid[nr, nc] != 0 and grid[nr, nc] != agent_color:
            break
        result[nr, nc] = agent_color
        r, c = nr, nc
        step_count += 1
        if steps > 0 and step_count >= steps:
            break

    return result


# ──────────────────────────────────────────────────────────────────────
# Object Transport (AGENTS + OBJECTS cross-system)
# ──────────────────────────────────────────────────────────────────────

def _transport_object(
    grid: np.ndarray, cargo_color: int, agent_color: int, goal_color: int
) -> np.ndarray:
    """
    Transport an object (cargo) toward a goal location.
    The cargo object moves to the goal position.
    This is a cross-system (AGENTS+OBJECTS) primitive.
    """
    result = grid.copy()
    cargo_cells = _find_all_colored_cells(grid, cargo_color)
    goal_pos = _find_colored_cell(grid, goal_color)
    if not cargo_cells or goal_pos is None:
        return result

    # Compute centroid of cargo
    cr = sum(r for r, _ in cargo_cells) / len(cargo_cells)
    cc = sum(c for _, c in cargo_cells) / len(cargo_cells)

    # Compute displacement to goal
    dr = goal_pos[0] - int(cr)
    dc = goal_pos[1] - int(cc)

    # Clear cargo
    for r, c in cargo_cells:
        result[r, c] = 0

    # Place cargo at new position
    h, w = grid.shape
    for r, c in cargo_cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result[nr, nc] = cargo_color

    return result


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_agent_primitives(registry: PrimitiveRegistry) -> None:
    """Register all AGENTS core knowledge primitives."""
    A = SpelkeSystem.AGENTS

    primitives = [
        # ── Pathfinding ──
        Primitive(
            "trace_path", Arrow(tgrid, Arrow(tcolor, Arrow(tcolor, tgrid))),
            lambda g: lambda ac: lambda gc: _draw_path(g, ac, gc),
            A, "Draw shortest path from agent color to goal color on grid",
            spelke_properties=["goal-directedness", "rational-efficiency"],
        ),
        Primitive(
            "move_toward", Arrow(tgrid, Arrow(tcolor, Arrow(tcolor, tgrid))),
            lambda g: lambda ac: lambda gc: _move_toward(g, ac, gc),
            A, "Move agent one step toward goal (rational efficiency)",
            spelke_properties=["equifinality", "rational-efficiency"],
        ),
        Primitive(
            "move_to_goal", Arrow(tgrid, Arrow(tcolor, Arrow(tcolor, tgrid))),
            lambda g: lambda ac: lambda gc: _move_to_goal(g, ac, gc),
            A, "Move agent directly to goal position",
            spelke_properties=["goal-directedness"],
        ),

        # ── Goal inference ──
        Primitive(
            "infer_goal_dir", Arrow(tgrid, Arrow(tcolor, tint)),
            lambda g: lambda ac: _infer_goal_direction(g, ac),
            A, "Infer likely goal direction from agent position (0=up,1=down,2=left,3=right)",
            spelke_properties=["goal-inference"],
        ),
        Primitive(
            "is_rational_path", Arrow(tgrid, Arrow(tcolor, Arrow(tcolor, Arrow(tcolor, tbool)))),
            lambda g: lambda pc: lambda ac: lambda gc: _is_rational_path(g, pc, ac, gc),
            A, "Check if a path is the shortest (rational efficiency criterion)",
            spelke_properties=["rational-efficiency"],
        ),

        # ── Trajectory projection ──
        Primitive(
            "project_trajectory", Arrow(tgrid, Arrow(tcolor, Arrow(tint, tgrid))),
            lambda g: lambda ac: lambda d: _project_trajectory(g, ac, d),
            A, "Project agent trajectory in given direction until obstacle",
            spelke_properties=["goal-directedness"],
        ),

        # ── Object transport (AGENTS+OBJECTS cross-system) ──
        Primitive(
            "transport_object",
            Arrow(tgrid, Arrow(tcolor, Arrow(tcolor, Arrow(tcolor, tgrid)))),
            lambda g: lambda cc: lambda ac: lambda gc: _transport_object(g, cc, ac, gc),
            A, "Transport cargo object toward goal (AGENTS+OBJECTS cross-system)",
            spelke_properties=["goal-directedness", "object-transport"],
        ),

        # ── Direction constants ──
        Primitive("dir_up", tint, 0, A, "Direction: up"),
        Primitive("dir_down", tint, 1, A, "Direction: down"),
        Primitive("dir_left", tint, 2, A, "Direction: left"),
        Primitive("dir_right", tint, 3, A, "Direction: right"),
    ]

    for p in primitives:
        registry.register(p)
"""

AGENTS module primitives: 11 total
- 3 movement (trace_path, move_toward, move_to_goal)
- 2 goal inference (infer_goal_dir, is_rational_path)
- 1 trajectory (project_trajectory)
- 1 transport (transport_object) — AGENTS+OBJECTS cross-system
- 4 direction constants
"""
