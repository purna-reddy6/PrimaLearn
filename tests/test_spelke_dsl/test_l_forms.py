"""Tests for the Spelke DSL — Forms/Geometry core system."""
import numpy as np
import pytest
from src.spelke_dsl.base import PrimitiveRegistry
from src.spelke_dsl.l_forms import (
    register_form_primitives,
    _has_horizontal_symmetry, _has_vertical_symmetry,
    _has_rotational_symmetry_90, _has_rotational_symmetry_180,
    _is_periodic, _find_period, _scale_up, _scale_down,
    _tile_grid, _rotate_grid_90, _rotate_n,
)


@pytest.fixture
def registry():
    reg = PrimitiveRegistry()
    register_form_primitives(reg)
    return reg


class TestSymmetry:
    def test_horizontal_symmetry_true(self):
        grid = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=np.int8)
        assert _has_horizontal_symmetry(grid)

    def test_horizontal_symmetry_false(self):
        grid = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.int8)
        assert not _has_horizontal_symmetry(grid)

    def test_vertical_symmetry(self):
        grid = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=np.int8)
        assert _has_vertical_symmetry(grid)

    def test_rotational_90(self):
        grid = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=np.int8)
        assert _has_rotational_symmetry_90(grid)

    def test_rotational_180(self):
        grid = np.array([
            [1, 2],
            [2, 1],
        ], dtype=np.int8)
        assert _has_rotational_symmetry_180(grid)


class TestTiling:
    def test_tile(self):
        pattern = np.array([[1, 2]], dtype=np.int8)
        tiled = _tile_grid(pattern, 2, 3)
        assert tiled.shape == (2, 6)

    def test_is_periodic_true(self):
        grid = np.array([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ], dtype=np.int8)
        assert _is_periodic(grid)

    def test_is_periodic_false(self):
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ], dtype=np.int8)
        assert not _is_periodic(grid)

    def test_find_period(self):
        grid = np.array([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ], dtype=np.int8)
        assert _find_period(grid) == (2, 2)


class TestScaling:
    def test_scale_up(self):
        grid = np.array([[1, 2]], dtype=np.int8)
        scaled = _scale_up(grid, 2)
        assert scaled.shape == (2, 4)
        assert scaled[0, 0] == 1
        assert scaled[0, 1] == 1
        assert scaled[0, 2] == 2

    def test_scale_down(self):
        grid = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int8)
        scaled = _scale_down(grid, 2)
        assert scaled.shape == (1, 2)
        assert scaled[0, 0] == 1
        assert scaled[0, 1] == 2


class TestTransformations:
    def test_rotate90(self):
        grid = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.int8)
        rotated = _rotate_grid_90(grid)
        assert rotated[0, 0] == 3
        assert rotated[0, 1] == 1
        assert rotated[1, 0] == 4
        assert rotated[1, 1] == 2


class TestRegistration:
    def test_count(self, registry):
        assert len(registry) >= 25

    def test_rotate_callable(self, registry):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        rot = registry["rotate90"]
        result = rot(grid)
        assert result.shape == (2, 2)


class TestRotateN:
    def test_n0_identity(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        assert np.array_equal(_rotate_n(grid, 0), grid)

    def test_n1_equals_rotate90(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        assert np.array_equal(_rotate_n(grid, 1), _rotate_grid_90(grid))

    def test_n2_equals_180(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        expected = np.array([[4, 3], [2, 1]], dtype=np.int8)
        assert np.array_equal(_rotate_n(grid, 2), expected)

    def test_n4_wraps_to_identity(self):
        grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        assert np.array_equal(_rotate_n(grid, 4), grid)

    def test_registered(self, registry):
        assert 'rotate_n' in registry
