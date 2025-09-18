"""Tests for the Spelke DSL — Agents core system."""
import numpy as np
import pytest
from src.spelke_dsl.base import PrimitiveRegistry, SpelkeSystem
from src.spelke_dsl.l_agents import (
    register_agent_primitives,
    _find_colored_cell, _find_all_colored_cells,
    _manhattan_distance, _trace_path, _draw_path,
    _move_toward, _move_to_goal,
    _infer_goal_direction, _is_rational_path,
    _project_trajectory, _transport_object,
)


@pytest.fixture
def registry():
    reg = PrimitiveRegistry()
    register_agent_primitives(reg)
    return reg


@pytest.fixture
def pathfinding_grid():
    """5x5 grid: agent (color 1) at (0,0), goal (color 2) at (4,4), obstacle (3) in middle."""
    return np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 2],
    ], dtype=np.int8)


@pytest.fixture
def simple_grid():
    """3x3 grid: agent (1) at (0,0), goal (2) at (2,2)."""
    return np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ], dtype=np.int8)


class TestColorCellFinding:
    def test_find_colored_cell(self, simple_grid):
        pos = _find_colored_cell(simple_grid, 1)
        assert pos == (0, 0)

    def test_find_missing_color(self, simple_grid):
        pos = _find_colored_cell(simple_grid, 5)
        assert pos is None

    def test_find_all_colored_cells(self, pathfinding_grid):
        cells = _find_all_colored_cells(pathfinding_grid, 3)
        assert len(cells) == 3  # Three obstacle cells


class TestPathfinding:
    def test_trace_simple_path(self, simple_grid):
        path = _trace_path(simple_grid, 1, 2)
        assert len(path) > 0
        assert path[0] == (0, 0)  # Start
        assert path[-1] == (2, 2)  # Goal

    def test_trace_path_around_obstacle(self, pathfinding_grid):
        path = _trace_path(pathfinding_grid, 1, 2)
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        # Path should NOT go through obstacles
        for r, c in path:
            if (r, c) != (0, 0) and (r, c) != (4, 4):
                assert pathfinding_grid[r, c] != 3

    def test_draw_path(self, simple_grid):
        result = _draw_path(simple_grid, 1, 2)
        # Path cells should be marked with agent color
        assert result[0, 0] == 1  # Agent stays
        assert result[2, 2] == 2  # Goal stays
        # At least some cells in between should be colored
        nonzero = np.count_nonzero(result)
        assert nonzero >= 3  # Agent + goal + at least 1 path cell


class TestMovement:
    def test_move_toward(self, simple_grid):
        result = _move_toward(simple_grid, 1, 2)
        # Agent should have moved from (0,0)
        assert result[0, 0] == 0  # Old position cleared
        # New position should be one step closer to goal

    def test_move_to_goal(self, simple_grid):
        result = _move_to_goal(simple_grid, 1, 2)
        # Agent should be at goal position
        assert result[0, 0] == 0  # Old position cleared
        assert result[2, 2] == 1  # Agent at goal (replaces goal)


class TestGoalInference:
    def test_infer_direction(self):
        # Agent at top-left, lots of structure at bottom-right
        grid = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 3, 3],
            [0, 0, 3, 3],
        ], dtype=np.int8)
        direction = _infer_goal_direction(grid, 1)
        # Should infer "down" or "right" (where the structure is)
        assert direction in [1, 3]  # 1=down, 3=right


class TestTrajectory:
    def test_project_right(self):
        grid = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int8)
        result = _project_trajectory(grid, 1, 3)  # 3=right
        # Should extend the agent to the right
        assert result[1, 0] == 1
        assert result[1, 1] == 1
        assert result[1, 4] == 1

    def test_project_stops_at_obstacle(self):
        grid = np.array([
            [1, 0, 0, 3, 0],
        ], dtype=np.int8)
        result = _project_trajectory(grid, 1, 3)  # 3=right
        # Should stop before obstacle
        assert result[0, 0] == 1
        assert result[0, 1] == 1
        assert result[0, 2] == 1
        assert result[0, 3] == 3  # Obstacle unchanged
        assert result[0, 4] == 0  # Beyond obstacle untouched


class TestTransport:
    def test_transport_object(self):
        grid = np.array([
            [4, 4, 0, 0, 0],
            [4, 4, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2],
        ], dtype=np.int8)
        result = _transport_object(grid, 4, 1, 2)
        # Cargo (4) should move toward goal (2)
        # Original position should be cleared
        assert result[0, 0] == 0


class TestRegistration:
    def test_registration_count(self, registry):
        assert len(registry) >= 10

    def test_all_tagged_agents(self, registry):
        for prim in registry:
            assert prim.system == SpelkeSystem.AGENTS

    def test_trace_path_callable(self, registry, simple_grid):
        trace = registry["trace_path"]
        result = trace(simple_grid)(1)(2)
        assert isinstance(result, np.ndarray)

    def test_direction_constants(self, registry):
        assert registry["dir_up"].implementation == 0
        assert registry["dir_down"].implementation == 1
        assert registry["dir_left"].implementation == 2
        assert registry["dir_right"].implementation == 3
