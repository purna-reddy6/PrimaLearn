"""Tests for the Spelke DSL — Places core system."""
import numpy as np
import pytest
from src.spelke_dsl.base import PrimitiveRegistry, SpelkeSystem
from src.spelke_dsl.l_places import (
    register_place_primitives,
    _extract_quadrant, _get_center_region, _get_border_region,
    _split_horizontal, _split_vertical,
    _get_row, _get_col,
    _distance_from_edge, _distance_from_center, _distance_from_color,
    _segment_by_axes, _reflect_across_axis, _reorient,
    _place_in_quadrant_8x8,
)


@pytest.fixture
def registry():
    reg = PrimitiveRegistry()
    register_place_primitives(reg)
    return reg


@pytest.fixture
def grid_4x4():
    """4x4 grid with distinct quadrants."""
    return np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.int8)


@pytest.fixture
def grid_5x5():
    """5x5 grid with content."""
    return np.array([
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [3, 0, 0, 0, 4],
    ], dtype=np.int8)


class TestQuadrantExtraction:
    def test_quadrant_tl(self, grid_4x4):
        q = _extract_quadrant(grid_4x4, 0)
        assert q.shape == (2, 2)
        assert np.all(q == 1)

    def test_quadrant_tr(self, grid_4x4):
        q = _extract_quadrant(grid_4x4, 1)
        assert q.shape == (2, 2)
        assert np.all(q == 2)

    def test_quadrant_bl(self, grid_4x4):
        q = _extract_quadrant(grid_4x4, 2)
        assert q.shape == (2, 2)
        assert np.all(q == 3)

    def test_quadrant_br(self, grid_4x4):
        q = _extract_quadrant(grid_4x4, 3)
        assert q.shape == (2, 2)
        assert np.all(q == 4)


class TestRegionExtraction:
    def test_center_region(self, grid_5x5):
        center = _get_center_region(grid_5x5, margin=1)
        assert center.shape == (3, 3)
        assert center[1, 1] == 5  # Center cell

    def test_border_region(self, grid_5x5):
        border = _get_border_region(grid_5x5)
        assert border[0, 0] == 1  # Top-left corner
        assert border[0, 4] == 2  # Top-right corner
        assert border[4, 0] == 3  # Bottom-left corner
        assert border[4, 4] == 4  # Bottom-right corner
        assert border[2, 2] == 0  # Center is not border

    def test_get_row(self, grid_5x5):
        row = _get_row(grid_5x5, 0)
        assert row.shape == (1, 5)
        assert row[0, 0] == 1

    def test_get_col(self, grid_5x5):
        col = _get_col(grid_5x5, 0)
        assert col.shape == (5, 1)
        assert col[0, 0] == 1
        assert col[4, 0] == 3


class TestDistanceFields:
    def test_distance_from_edge(self):
        grid = np.zeros((5, 5), dtype=np.int8)
        dist = _distance_from_edge(grid)
        assert dist[0, 0] == 0  # Corner
        assert dist[2, 2] == 2  # Center
        assert dist[1, 1] == 1  # One step in

    def test_distance_from_center(self):
        grid = np.zeros((5, 5), dtype=np.int8)
        dist = _distance_from_center(grid)
        assert dist[2, 2] == 0  # Center
        assert dist[0, 0] == 4  # Corner (2+2)
        assert dist[0, 2] == 2  # Top center (2+0)

    def test_distance_from_color(self):
        grid = np.array([
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        dist = _distance_from_color(grid, 5)
        assert dist[1, 1] == 0  # The color cell itself
        assert dist[0, 1] == 1  # Adjacent
        assert dist[0, 0] == 2  # Diagonal


class TestSpatialSegmentation:
    def test_segment_by_axes(self):
        grid = np.zeros((5, 5), dtype=np.int8)
        seg = _segment_by_axes(grid)
        assert seg[0, 0] == 1  # TL
        assert seg[0, 4] == 2  # TR
        assert seg[4, 0] == 3  # BL
        assert seg[4, 4] == 4  # BR
        assert seg[2, 2] == 0  # Center axis

    def test_reflect_across_h(self):
        grid = np.array([
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        result = _reflect_across_axis(grid, 0)
        assert result[2, 0] == 1  # Reflected
        assert result[2, 1] == 1

    def test_reflect_across_v(self):
        grid = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        result = _reflect_across_axis(grid, 1)
        assert result[0, 2] == 1  # Reflected
        assert result[1, 2] == 1


class TestReorientation:
    def test_reorient_already_tl(self, grid_5x5):
        """Grid with most content in TL should stay same."""
        grid = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        result = _reorient(grid)
        assert np.array_equal(result, grid)

    def test_reorient_from_br(self):
        """Grid with most content in BR should be rotated 180."""
        grid = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=np.int8)
        result = _reorient(grid)
        # After 180 rotation, the dense quadrant should be TL
        assert np.count_nonzero(result[:2, :2]) >= np.count_nonzero(result[2:, 2:])


class TestRegistration:
    def test_registration_count(self, registry):
        assert len(registry) >= 14

    def test_all_tagged_places(self, registry):
        for prim in registry:
            assert prim.system == SpelkeSystem.PLACES

    def test_extract_quadrant_callable(self, registry, grid_4x4):
        ext = registry["extract_quadrant"]
        result = ext(grid_4x4)(0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_quadrant_constants(self, registry):
        assert registry["quad_tl"].implementation == 0
        assert registry["quad_tr"].implementation == 1
        assert registry["quad_bl"].implementation == 2
        assert registry["quad_br"].implementation == 3


class TestPlaceInQuadrant8x8:
    """Unit tests for the FORMS+OBJECTS+PLACES bridge: place_in_quadrant_8x8."""

    def test_output_is_8x8(self):
        small = np.array([[1, 2], [3, 4]], dtype=np.int8)
        result = _place_in_quadrant_8x8(small, 0)
        assert result.shape == (8, 8)

    def test_place_top_left(self):
        small = np.array([[5, 6]], dtype=np.int8)  # 1x2
        result = _place_in_quadrant_8x8(small, 0)  # quad_tl
        assert result[0, 0] == 5
        assert result[0, 1] == 6
        assert result[0, 4] == 0  # TR quadrant empty
        assert result[4, 0] == 0  # BL quadrant empty

    def test_place_top_right(self):
        small = np.array([[7, 8]], dtype=np.int8)  # 1x2
        result = _place_in_quadrant_8x8(small, 1)  # quad_tr
        assert result[0, 4] == 7
        assert result[0, 5] == 8
        assert result[0, 0] == 0  # TL quadrant empty

    def test_place_bottom_left(self):
        small = np.array([[3]], dtype=np.int8)  # 1x1
        result = _place_in_quadrant_8x8(small, 2)  # quad_bl
        assert result[4, 0] == 3
        assert result[0, 0] == 0

    def test_place_bottom_right(self):
        small = np.array([[9]], dtype=np.int8)  # 1x1
        result = _place_in_quadrant_8x8(small, 3)  # quad_br
        assert result[4, 4] == 9
        assert result[0, 0] == 0

    def test_canvas_zeros_elsewhere(self):
        small = np.array([[1, 2], [3, 4]], dtype=np.int8)
        result = _place_in_quadrant_8x8(small, 1)  # TR
        # TL quadrant should be zero
        assert (result[:4, :4] == 0).all()
        # BL quadrant should be zero
        assert (result[4:, :4] == 0).all()
        # BR quadrant should be zero
        assert (result[4:, 4:] == 0).all()
        # TR: rows 0-1, cols 4-5 should have small content
        assert result[0, 4] == 1
        assert result[0, 5] == 2

    def test_registered_in_registry(self):
        reg = PrimitiveRegistry()
        register_place_primitives(reg)
        assert "place_in_quadrant_8x8" in reg._primitives
