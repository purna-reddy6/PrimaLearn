"""Tests for the Spelke DSL — PERSONS core system."""
import numpy as np
import pytest
from src.spelke_dsl.base import PrimitiveRegistry, SpelkeSystem
from src.spelke_dsl.l_persons import (
    register_persons_primitives,
    _is_agent_obj, _is_target_obj, _agent_count,
    _obj_facing_dir, _point_toward, _social_dist,
    _extract_agents, _extract_targets, _nearest_agent, _nearest_target,
    _goal_reachable, _mirror_intent,
    _aspect_ratio, _centroid, _obj_color, _extract_objects_list,
)


@pytest.fixture
def registry():
    reg = PrimitiveRegistry()
    register_persons_primitives(reg)
    return reg


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def tall_agent():
    """3×1 tall object — agent-like (aspect ratio 3.0)."""
    return np.array([[1], [1], [1]], dtype=np.int8)


@pytest.fixture
def wide_agent():
    """1×3 wide object — agent-like (aspect ratio 3.0)."""
    return np.array([[2, 2, 2]], dtype=np.int8)


@pytest.fixture
def square_target():
    """2×2 compact object — target-like (aspect ratio 1.0, area 4)."""
    return np.array([[3, 3], [3, 3]], dtype=np.int8)


@pytest.fixture
def tiny_obj():
    """1×1 single cell — NOT a target (area <= 2)."""
    return np.array([[4]], dtype=np.int8)


@pytest.fixture
def mixed_grid():
    """10×10 grid with 1 agent (1×3 row of color 1) and 1 target (2×2 block of color 2)."""
    g = np.zeros((10, 10), dtype=np.int8)
    # Agent: row at (2, 1-3)
    g[2, 1] = 1
    g[2, 2] = 1
    g[2, 3] = 1
    # Target: 2×2 at (7, 7-8)
    g[7, 7] = 2
    g[7, 8] = 2
    g[8, 7] = 2
    g[8, 8] = 2
    return g


@pytest.fixture
def multi_agent_grid():
    """Grid with 2 agents (color 1 = 1×3, color 3 = 4×1) and 1 target (2×2 color 5)."""
    g = np.zeros((12, 12), dtype=np.int8)
    # Agent 1: 1×3 wide row
    g[1, 1] = 1; g[1, 2] = 1; g[1, 3] = 1
    # Agent 2: 4×1 tall column
    g[4, 9] = 3; g[5, 9] = 3; g[6, 9] = 3; g[7, 9] = 3
    # Target: 2×2 block
    g[9, 5] = 5; g[9, 6] = 5; g[10, 5] = 5; g[10, 6] = 5
    return g


@pytest.fixture
def reachable_grid():
    """Agent (color 1, 1×3) can reach target (color 2, 2×2) with no obstacles."""
    g = np.zeros((8, 8), dtype=np.int8)
    g[1, 1] = 1; g[1, 2] = 1; g[1, 3] = 1
    g[5, 5] = 2; g[5, 6] = 2; g[6, 5] = 2; g[6, 6] = 2
    return g


@pytest.fixture
def blocked_grid():
    """Agent blocked from target by wall of color 9."""
    g = np.zeros((8, 8), dtype=np.int8)
    g[3, 1] = 1; g[3, 2] = 1; g[3, 3] = 1
    # Wall
    for r in range(8):
        g[r, 5] = 9
    g[6, 6] = 2; g[6, 7] = 2; g[7, 6] = 2; g[7, 7] = 2
    return g


# ──────────────────────────────────────────────────────────────────────
# Registry Tests
# ──────────────────────────────────────────────────────────────────────

class TestRegistration:
    def test_register_all(self, registry):
        assert len(registry) == 12

    def test_all_persons_system(self, registry):
        for p in registry:
            assert p.system == SpelkeSystem.PERSONS

    def test_primitive_names(self, registry):
        names = registry.names()
        expected = [
            "is_agent_obj", "is_target_obj", "agent_count",
            "obj_facing_dir", "point_toward", "social_dist",
            "extract_agents", "extract_targets", "nearest_agent", "nearest_target",
            "goal_reachable", "mirror_intent",
        ]
        for name in expected:
            assert name in names, f"Missing: {name}"

    def test_no_duplicate_registration(self):
        reg = PrimitiveRegistry()
        register_persons_primitives(reg)
        with pytest.raises(ValueError):
            register_persons_primitives(reg)


# ──────────────────────────────────────────────────────────────────────
# Helper Tests
# ──────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_aspect_ratio_tall(self, tall_agent):
        assert _aspect_ratio(tall_agent) == pytest.approx(3.0)

    def test_aspect_ratio_wide(self, wide_agent):
        assert _aspect_ratio(wide_agent) == pytest.approx(3.0)

    def test_aspect_ratio_square(self, square_target):
        assert _aspect_ratio(square_target) == pytest.approx(1.0)

    def test_aspect_ratio_empty(self):
        assert _aspect_ratio(np.zeros((0, 0), dtype=np.int8)) == 1.0

    def test_centroid_single(self, tiny_obj):
        r, c = _centroid(tiny_obj)
        assert r == 0.0 and c == 0.0

    def test_centroid_2x2(self, square_target):
        r, c = _centroid(square_target)
        assert r == pytest.approx(0.5)
        assert c == pytest.approx(0.5)

    def test_centroid_empty(self):
        r, c = _centroid(np.zeros((3, 3), dtype=np.int8))
        assert r == 0.0 and c == 0.0

    def test_obj_color(self, tall_agent):
        assert _obj_color(tall_agent) == 1

    def test_obj_color_empty(self):
        assert _obj_color(np.zeros((3, 3), dtype=np.int8)) == 0

    def test_extract_objects_list_two_objects(self, mixed_grid):
        objs = _extract_objects_list(mixed_grid)
        assert len(objs) == 2

    def test_extract_objects_list_empty(self):
        assert _extract_objects_list(np.zeros((5, 5), dtype=np.int8)) == []


# ──────────────────────────────────────────────────────────────────────
# Detection Primitive Tests
# ──────────────────────────────────────────────────────────────────────

class TestIsAgentObj:
    def test_tall_is_agent(self, tall_agent):
        assert _is_agent_obj(tall_agent) is True

    def test_wide_is_agent(self, wide_agent):
        assert _is_agent_obj(wide_agent) is True

    def test_square_not_agent(self, square_target):
        assert _is_agent_obj(square_target) is False

    def test_empty_not_agent(self):
        assert _is_agent_obj(np.zeros((0, 0), dtype=np.int8)) is False

    def test_1x2_borderline(self):
        # 1×2: ratio 2.0 → agent
        obj = np.array([[1, 1]], dtype=np.int8)
        assert _is_agent_obj(obj) is True

    def test_2x2_not_agent(self):
        obj = np.array([[1, 1], [1, 1]], dtype=np.int8)
        assert _is_agent_obj(obj) is False

    def test_1x5_agent(self):
        obj = np.array([[1, 1, 1, 1, 1]], dtype=np.int8)
        assert _is_agent_obj(obj) is True


class TestIsTargetObj:
    def test_square_is_target(self, square_target):
        assert _is_target_obj(square_target) is True

    def test_tiny_not_target(self, tiny_obj):
        assert _is_target_obj(tiny_obj) is False  # area = 1, not > 2

    def test_agent_not_target(self, tall_agent):
        assert _is_target_obj(tall_agent) is False

    def test_3x3_is_target(self):
        obj = np.ones((3, 3), dtype=np.int8) * 2
        assert _is_target_obj(obj) is True

    def test_empty_not_target(self):
        assert _is_target_obj(np.zeros((0, 0), dtype=np.int8)) is False

    def test_2x3_borderline(self):
        # 2×3: ratio = 1.5, which is NOT < 1.3 → not target
        obj = np.ones((2, 3), dtype=np.int8)
        assert _is_target_obj(obj) is False


class TestAgentCount:
    def test_mixed_grid_one_agent(self, mixed_grid):
        assert _agent_count(mixed_grid) == 1

    def test_multi_agent_grid(self, multi_agent_grid):
        assert _agent_count(multi_agent_grid) == 2

    def test_empty_grid(self):
        assert _agent_count(np.zeros((5, 5), dtype=np.int8)) == 0

    def test_only_targets(self):
        g = np.zeros((8, 8), dtype=np.int8)
        g[2:4, 2:4] = 3  # 2×2 target
        assert _agent_count(g) == 0


# ──────────────────────────────────────────────────────────────────────
# Trajectory / Intentionality Primitive Tests
# ──────────────────────────────────────────────────────────────────────

class TestObjFacingDir:
    def test_tall_faces_up(self, tall_agent):
        assert _obj_facing_dir(tall_agent) == 0

    def test_wide_faces_right(self, wide_agent):
        assert _obj_facing_dir(wide_agent) == 1

    def test_square_defaults_up(self, square_target):
        assert _obj_facing_dir(square_target) == 0

    def test_empty_defaults_up(self):
        assert _obj_facing_dir(np.zeros((0, 0), dtype=np.int8)) == 0

    def test_1x4_faces_right(self):
        obj = np.array([[1, 1, 1, 1]], dtype=np.int8)
        assert _obj_facing_dir(obj) == 1

    def test_4x1_faces_up(self):
        obj = np.array([[1], [1], [1], [1]], dtype=np.int8)
        assert _obj_facing_dir(obj) == 0


class TestPointToward:
    def test_returns_grid_same_size(self, mixed_grid):
        objs = _extract_objects_list(mixed_grid)
        if len(objs) >= 2:
            result = _point_toward(objs[0], objs[1], mixed_grid)
            assert result.shape == mixed_grid.shape

    def test_returns_grid_on_empty(self):
        g = np.zeros((5, 5), dtype=np.int8)
        result = _point_toward(g, g, g)
        assert result.shape == (5, 5)

    def test_no_change_already_aligned(self, mixed_grid):
        # wide agent already facing right, target is to its right/below
        # result should still be valid grid
        agent = _nearest_agent(mixed_grid)
        target = _nearest_target(mixed_grid, agent)
        result = _point_toward(agent, target, mixed_grid)
        assert result.shape == mixed_grid.shape


class TestSocialDist:
    def test_zero_same_object(self, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        dist = _social_dist(agent, agent, mixed_grid)
        assert dist == 0

    def test_positive_different_objects(self, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        target = _nearest_target(mixed_grid, agent)
        dist = _social_dist(agent, target, mixed_grid)
        assert dist > 0

    def test_empty_grid(self):
        g = np.zeros((5, 5), dtype=np.int8)
        assert _social_dist(g, g, g) == 0

    def test_social_dist_symmetry(self, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        target = _nearest_target(mixed_grid, agent)
        d1 = _social_dist(agent, target, mixed_grid)
        d2 = _social_dist(target, agent, mixed_grid)
        assert d1 == d2


# ──────────────────────────────────────────────────────────────────────
# Social Composition Primitive Tests
# ──────────────────────────────────────────────────────────────────────

class TestExtractAgents:
    def test_extracts_agent_only(self, mixed_grid):
        agents = _extract_agents(mixed_grid)
        # Agent (color 1) should be present, target (color 2) absent
        assert np.any(agents == 1)
        assert not np.any(agents == 2)

    def test_empty_grid(self):
        g = np.zeros((5, 5), dtype=np.int8)
        result = _extract_agents(g)
        assert np.all(result == 0)

    def test_two_agents(self, multi_agent_grid):
        agents = _extract_agents(multi_agent_grid)
        unique = set(agents[agents != 0].tolist())
        assert len(unique) == 2

    def test_target_not_in_agents(self, multi_agent_grid):
        agents = _extract_agents(multi_agent_grid)
        assert not np.any(agents == 5)


class TestExtractTargets:
    def test_extracts_target_only(self, mixed_grid):
        targets = _extract_targets(mixed_grid)
        assert np.any(targets == 2)
        assert not np.any(targets == 1)

    def test_empty_grid(self):
        g = np.zeros((5, 5), dtype=np.int8)
        result = _extract_targets(g)
        assert np.all(result == 0)

    def test_agents_not_in_targets(self, multi_agent_grid):
        targets = _extract_targets(multi_agent_grid)
        assert not np.any(targets == 1)
        assert not np.any(targets == 3)


class TestNearestAgent:
    def test_finds_agent_in_mixed_grid(self, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        assert np.any(agent == 1)
        assert not np.any(agent == 2)

    def test_empty_grid_returns_zeros(self):
        g = np.zeros((5, 5), dtype=np.int8)
        result = _nearest_agent(g)
        assert np.all(result == 0)

    def test_only_one_agent_returned(self, multi_agent_grid):
        agent = _nearest_agent(multi_agent_grid)
        # Should return exactly one connected component (largest)
        unique_colors = set(agent[agent != 0].tolist())
        assert len(unique_colors) == 1

    def test_largest_agent_selected(self, multi_agent_grid):
        # Agent 2 (4×1 = 4 cells) vs Agent 1 (1×3 = 3 cells) → Agent 2 (color 3)
        agent = _nearest_agent(multi_agent_grid)
        assert np.any(agent == 3)


class TestNearestTarget:
    def test_finds_target_for_agent(self, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        target = _nearest_target(mixed_grid, agent)
        assert np.any(target == 2)

    def test_empty_grid_returns_zeros(self):
        g = np.zeros((5, 5), dtype=np.int8)
        result = _nearest_target(g, g)
        assert np.all(result == 0)

    def test_target_not_agent(self, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        target = _nearest_target(mixed_grid, agent)
        # Target should not overlap with agent color
        assert not np.any(target == 1)


# ──────────────────────────────────────────────────────────────────────
# Theory of Mind Primitive Tests
# ──────────────────────────────────────────────────────────────────────

class TestGoalReachable:
    def test_reachable_open_field(self, reachable_grid):
        agent = _nearest_agent(reachable_grid)
        target = _nearest_target(reachable_grid, agent)
        assert _goal_reachable(agent, target, reachable_grid) is True

    def test_blocked_by_wall(self):
        # Directly construct agent/target masks to bypass agent-detection pipeline
        # (an 8×1 wall column is elongated and would be misidentified as agent)
        g = np.zeros((8, 8), dtype=np.int8)
        g[3, 1] = 1; g[3, 2] = 1; g[3, 3] = 1  # agent row
        for r in range(8):
            g[r, 5] = 9  # wall at column 5
        g[6, 6] = 2; g[6, 7] = 2; g[7, 6] = 2; g[7, 7] = 2
        # Build masks manually
        agent_mask = np.zeros_like(g)
        agent_mask[3, 1] = 1; agent_mask[3, 2] = 1; agent_mask[3, 3] = 1
        target_mask = np.zeros_like(g)
        target_mask[6, 6] = 2; target_mask[6, 7] = 2
        target_mask[7, 6] = 2; target_mask[7, 7] = 2
        assert _goal_reachable(agent_mask, target_mask, g) is False

    def test_empty_agent_false(self):
        g = np.zeros((5, 5), dtype=np.int8)
        assert _goal_reachable(g, g, g) is False

    def test_adjacent_reachable(self):
        g = np.zeros((5, 5), dtype=np.int8)
        # Agent: 1×3 at row 2
        g[2, 0] = 1; g[2, 1] = 1; g[2, 2] = 1
        # Target: 2×2 adjacent
        g[2, 3] = 2; g[2, 4] = 2; g[3, 3] = 2; g[3, 4] = 2
        agent = _nearest_agent(g)
        target = _nearest_target(g, agent)
        assert _goal_reachable(agent, target, g) is True


class TestMirrorIntent:
    def test_wide_agent_flips_horizontal(self):
        g = np.zeros((4, 6), dtype=np.int8)
        # Wide agent (1×3)
        g[1, 1] = 1; g[1, 2] = 1; g[1, 3] = 1
        g[0, 5] = 7  # asymmetric marker
        result = _mirror_intent(g)
        assert result.shape == g.shape
        # Horizontal flip: marker at (0,5) moves to (0,0)
        assert result[0, 0] == 7

    def test_tall_agent_flips_vertical(self):
        g = np.zeros((6, 4), dtype=np.int8)
        # Tall agent (3×1)
        g[1, 1] = 1; g[2, 1] = 1; g[3, 1] = 1
        g[0, 3] = 7  # marker at top-right
        result = _mirror_intent(g)
        assert result.shape == g.shape
        # Vertical flip: marker at (0,3) moves to (5,3)
        assert result[5, 3] == 7

    def test_empty_grid_returns_flipped(self):
        g = np.zeros((4, 4), dtype=np.int8)
        result = _mirror_intent(g)
        assert result.shape == (4, 4)

    def test_idempotent_up_to_double(self):
        g = np.zeros((6, 4), dtype=np.int8)
        g[1, 1] = 1; g[2, 1] = 1; g[3, 1] = 1  # tall
        g[0, 0] = 5
        result = _mirror_intent(g)
        result2 = _mirror_intent(result)
        # Double mirror = identity (for vertical flip)
        assert np.array_equal(result2, g)


# ──────────────────────────────────────────────────────────────────────
# Callable Interface Tests (via registry)
# ──────────────────────────────────────────────────────────────────────

class TestCallableInterface:
    def test_is_agent_obj_callable(self, registry, tall_agent):
        p = registry.get("is_agent_obj")
        assert p(tall_agent) is True

    def test_is_target_obj_callable(self, registry, square_target):
        p = registry.get("is_target_obj")
        assert p(square_target) is True

    def test_agent_count_callable(self, registry, mixed_grid):
        p = registry.get("agent_count")
        assert p(mixed_grid) == 1

    def test_obj_facing_dir_callable(self, registry, tall_agent):
        p = registry.get("obj_facing_dir")
        assert p(tall_agent) == 0

    def test_social_dist_callable(self, registry, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        target = _nearest_target(mixed_grid, agent)
        p = registry.get("social_dist")
        dist = p(agent)(target)(mixed_grid)
        assert dist > 0

    def test_extract_agents_callable(self, registry, mixed_grid):
        p = registry.get("extract_agents")
        result = p(mixed_grid)
        assert np.any(result == 1)

    def test_nearest_agent_callable(self, registry, mixed_grid):
        p = registry.get("nearest_agent")
        result = p(mixed_grid)
        assert np.any(result == 1)

    def test_nearest_target_callable(self, registry, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        p = registry.get("nearest_target")
        result = p(mixed_grid)(agent)
        assert np.any(result == 2)

    def test_goal_reachable_callable(self, registry, reachable_grid):
        agent = _nearest_agent(reachable_grid)
        target = _nearest_target(reachable_grid, agent)
        p = registry.get("goal_reachable")
        assert p(agent)(target)(reachable_grid) is True

    def test_mirror_intent_callable(self, registry):
        g = np.zeros((4, 6), dtype=np.int8)
        g[1, 0] = 1; g[1, 1] = 1; g[1, 2] = 1  # wide agent
        p = registry.get("mirror_intent")
        result = p(g)
        assert result.shape == g.shape

    def test_point_toward_callable(self, registry, mixed_grid):
        agent = _nearest_agent(mixed_grid)
        target = _nearest_target(mixed_grid, agent)
        p = registry.get("point_toward")
        result = p(agent)(target)(mixed_grid)
        assert result.shape == mixed_grid.shape

    def test_extract_targets_callable(self, registry, mixed_grid):
        p = registry.get("extract_targets")
        result = p(mixed_grid)
        assert np.any(result == 2)


# ──────────────────────────────────────────────────────────────────────
# Edge Case / Robustness Tests
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_primitives_handle_zero_grid(self, registry):
        g = np.zeros((5, 5), dtype=np.int8)
        # These return grids
        for name in ["extract_agents", "extract_targets", "nearest_agent", "mirror_intent"]:
            result = registry.get(name)(g)
            assert result.shape == (5, 5)

    def test_agent_count_single_cell_objects(self):
        g = np.zeros((5, 5), dtype=np.int8)
        g[1, 1] = 1
        g[3, 3] = 2
        # 1×1 cells — not elongated → 0 agents
        assert _agent_count(g) == 0

    def test_social_dist_missing_color(self, mixed_grid):
        fake_obj = np.zeros((3, 3), dtype=np.int8)
        result = _social_dist(fake_obj, fake_obj, mixed_grid)
        assert result == 0

    def test_is_agent_obj_1x1_false(self):
        obj = np.array([[5]], dtype=np.int8)
        assert _is_agent_obj(obj) is False  # aspect ratio = 1.0

    def test_is_target_obj_1x1_false(self):
        obj = np.array([[5]], dtype=np.int8)
        assert _is_target_obj(obj) is False  # area = 1 ≤ 2

    def test_nearest_target_no_targets(self):
        g = np.zeros((8, 8), dtype=np.int8)
        # Only an agent, no target
        g[2, 1] = 1; g[2, 2] = 1; g[2, 3] = 1
        agent = _nearest_agent(g)
        result = _nearest_target(g, agent)
        assert np.all(result == 0)

    def test_goal_reachable_no_target(self):
        g = np.zeros((8, 8), dtype=np.int8)
        g[2, 1] = 1; g[2, 2] = 1; g[2, 3] = 1
        agent = _nearest_agent(g)
        empty = np.zeros_like(g)
        assert _goal_reachable(agent, empty, g) is False

    def test_extract_objects_list_single(self, tall_agent):
        objs = _extract_objects_list(tall_agent)
        assert len(objs) == 1
        assert objs[0].shape[0] == 3 and objs[0].shape[1] == 1
