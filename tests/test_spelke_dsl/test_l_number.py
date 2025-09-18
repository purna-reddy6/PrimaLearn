"""Tests for the Spelke DSL — Number core system."""
import numpy as np
import pytest
from src.spelke_dsl.base import PrimitiveRegistry
from src.spelke_dsl.l_number import (
    register_number_primitives,
    _count_exact_small, _ans_greater, _ans_approximate_equal,
    _successor, _count, _count_cells, _render_count_colored,
    _count_in_quadrant,
)


@pytest.fixture
def registry():
    reg = PrimitiveRegistry()
    register_number_primitives(reg)
    return reg


class TestOTS:
    """Parallel individuation — Object Tracking System, cap ≤3."""

    def test_ots_one(self):
        assert _count_exact_small([1]) == 1

    def test_ots_two(self):
        assert _count_exact_small([1, 2]) == 2

    def test_ots_three(self):
        assert _count_exact_small([1, 2, 3]) == 3

    def test_ots_beyond_cap(self):
        """OTS cannot handle >3 — returns -1 (signature limit)."""
        assert _count_exact_small([1, 2, 3, 4]) == -1
        assert _count_exact_small([1, 2, 3, 4, 5]) == -1

    def test_ots_empty(self):
        assert _count_exact_small([]) == 0


class TestANS:
    """Approximate Number System — ratio-dependent."""

    def test_ans_clearly_greater(self):
        assert _ans_greater(10, 5)

    def test_ans_clearly_less(self):
        assert not _ans_greater(5, 10)

    def test_ans_too_close(self):
        """ANS cannot distinguish numbers with ratio < 1 + Weber fraction."""
        assert not _ans_greater(100, 99)  # Ratio 1.01, below Weber ~0.15

    def test_ans_approx_equal_close(self):
        assert _ans_approximate_equal(100, 105)

    def test_ans_approx_equal_far(self):
        assert not _ans_approximate_equal(10, 20)


class TestExactArithmetic:
    """Bootstrapped exact arithmetic (cardinal principle)."""

    def test_count(self):
        assert _count([1, 2, 3, 4, 5]) == 5

    def test_successor(self):
        assert _successor(0) == 1
        assert _successor(4) == 5

    def test_constants(self, registry):
        assert registry["zero"]() == 0
        assert registry["one"]() == 1
        assert registry["nine"]() == 9

    def test_arithmetic(self, registry):
        add = registry["add"]
        assert add(3)(4) == 7

        sub = registry["sub"]
        assert sub(5)(3) == 2
        assert sub(3)(5) == 0  # Floored at 0


class TestRegistration:
    def test_count(self, registry):
        assert len(registry) >= 35  # OTS + ANS + arithmetic + constants

    def test_has_ots(self, registry):
        assert "ots_count" in registry

    def test_has_ans(self, registry):
        assert "ans_greater" in registry

    def test_has_successor(self, registry):
        assert "succ" in registry

    def test_has_count_cells(self, registry):
        assert "count_cells" in registry

    def test_count_cells_is_number_system(self, registry):
        from src.spelke_dsl.base import SpelkeSystem
        assert registry["count_cells"].system == SpelkeSystem.NUMBER


class TestCountCells:
    """Tests for count_cells bridge primitive."""

    def test_scattered_cells(self):
        """Non-adjacent single cells: count_cells == count_objects."""
        grid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert _count_cells(grid) == 3

    def test_connected_region(self):
        """Adjacent cells form one connected component but count_cells > 1."""
        grid = np.array([[2, 2], [2, 2]])
        assert _count_cells(grid) == 4

    def test_empty_grid(self):
        assert _count_cells(np.zeros((3, 3), dtype=int)) == 0

    def test_single_cell(self):
        grid = np.array([[0, 7, 0], [0, 0, 0]])
        assert _count_cells(grid) == 1

    def test_d631b094_examples(self):
        """Verify count_cells + render_count_colored solves d631b094 pattern."""
        examples = [
            # (n_cells, color, expected_output_shape)
            (2, 1, (1, 2)),
            (3, 2, (1, 3)),
            (1, 7, (1, 1)),
            (4, 8, (1, 4)),
            (5, 4, (1, 5)),
        ]
        for n, color, shape in examples:
            grid = np.zeros((3, 3), dtype=int)
            for i in range(n):
                grid.flat[i] = color
            assert _count_cells(grid) == n
            out = _render_count_colored(n, color)
            assert out.shape == shape
            assert np.all(out == color)


class TestCountInQuadrant:
    def test_empty_quadrant(self):
        grid = np.zeros((8, 8), dtype=int)
        grid[0, 0] = 1  # TL only
        assert _count_in_quadrant(grid, 1) == 0  # TR empty

    def test_one_object_per_quadrant(self):
        grid = np.zeros((8, 8), dtype=int)
        grid[1, 1] = 1   # TL
        grid[1, 5] = 2   # TR
        grid[5, 1] = 3   # BL
        grid[5, 5] = 4   # BR
        assert _count_in_quadrant(grid, 0) == 1
        assert _count_in_quadrant(grid, 1) == 1
        assert _count_in_quadrant(grid, 2) == 1
        assert _count_in_quadrant(grid, 3) == 1

    def test_multiple_objects_in_quadrant(self):
        grid = np.zeros((8, 8), dtype=int)
        grid[0, 0] = 1
        grid[1, 2] = 2
        grid[2, 0] = 3
        assert _count_in_quadrant(grid, 0) == 3  # three objects in TL

    def test_registered(self, registry):
        assert 'count_in_quadrant' in registry
