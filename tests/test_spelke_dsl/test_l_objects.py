"""Tests for the Spelke DSL — Objects core system."""
import numpy as np
import pytest
from src.spelke_dsl.base import PrimitiveRegistry
from src.spelke_dsl.l_objects import (
    register_object_primitives, _extract_objects, _touching,
    _reflect_object_h, _reflect_object_v, _rotate_object_90,
    GridObject,
)
from src.spelke_dsl.l_number import _render_count_colored, _tile_n


@pytest.fixture
def registry():
    reg = PrimitiveRegistry()
    register_object_primitives(reg)
    return reg


@pytest.fixture
def sample_grid():
    """3x3 grid with two objects."""
    return np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 2],
    ], dtype=np.int8)


@pytest.fixture
def complex_grid():
    """5x5 grid with three objects."""
    return np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 2, 0, 3, 3],
        [2, 2, 0, 3, 3],
    ], dtype=np.int8)


class TestObjectExtraction:
    def test_extract_basic(self, sample_grid):
        objs = _extract_objects(sample_grid)
        assert len(objs) == 2
        colors = sorted(o.color for o in objs)
        assert colors == [1, 2]

    def test_extract_sizes(self, sample_grid):
        objs = _extract_objects(sample_grid)
        sizes = sorted(o.size for o in objs)
        assert sizes == [1, 3]  # blue L-shape (3), red dot (1)

    def test_extract_complex(self, complex_grid):
        objs = _extract_objects(complex_grid)
        assert len(objs) == 3

    def test_empty_grid(self):
        grid = np.zeros((3, 3), dtype=np.int8)
        objs = _extract_objects(grid)
        assert len(objs) == 0

    def test_full_grid(self):
        grid = np.ones((3, 3), dtype=np.int8)
        objs = _extract_objects(grid)
        assert len(objs) == 1
        assert objs[0].size == 9

    def test_diagonal_not_connected_4conn(self):
        grid = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.int8)
        objs = _extract_objects(grid, connectivity=4)
        assert len(objs) == 2  # Diagonal cells are NOT connected in 4-conn

    def test_diagonal_connected_8conn(self):
        grid = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.int8)
        objs = _extract_objects(grid, connectivity=8)
        assert len(objs) == 1  # Diagonal cells ARE connected in 8-conn


class TestObjectProperties:
    def test_bbox(self, sample_grid):
        objs = _extract_objects(sample_grid)
        blue = [o for o in objs if o.color == 1][0]
        assert blue.bbox == (0, 0, 1, 1)  # (r_min, c_min, r_max, c_max)

    def test_center(self, sample_grid):
        objs = _extract_objects(sample_grid)
        red = [o for o in objs if o.color == 2][0]
        assert red.center == (2.0, 2.0)

    def test_to_grid(self, sample_grid):
        objs = _extract_objects(sample_grid)
        blue = [o for o in objs if o.color == 1][0]
        mini = blue.to_grid()
        assert mini.shape == (2, 2)
        assert mini[0, 0] == 1
        assert mini[0, 1] == 1
        assert mini[1, 0] == 1


class TestObjectTransformations:
    def test_translate(self, sample_grid):
        objs = _extract_objects(sample_grid)
        blue = [o for o in objs if o.color == 1][0]
        moved = blue.translate(1, 1)
        assert moved.color == 1
        assert (1, 1) in moved.cells
        assert (1, 2) in moved.cells

    def test_reflect_h(self, sample_grid):
        objs = _extract_objects(sample_grid)
        blue = [o for o in objs if o.color == 1][0]
        reflected = _reflect_object_h(blue)
        assert reflected.color == blue.color
        assert reflected.size == blue.size


class TestContactDetection:
    def test_touching(self):
        grid = np.array([
            [1, 2],
            [0, 0],
        ], dtype=np.int8)
        objs = _extract_objects(grid)
        assert len(objs) == 2
        assert _touching(objs[0], objs[1])

    def test_not_touching(self):
        grid = np.array([
            [1, 0, 2],
            [0, 0, 0],
        ], dtype=np.int8)
        objs = _extract_objects(grid)
        assert len(objs) == 2
        assert not _touching(objs[0], objs[1])


class TestRenderCountColored:
    """Unit tests for the NUMBER+OBJECTS bridge: render_count_colored."""

    def test_basic_count_3(self):
        result = _render_count_colored(3, 1)
        assert result.shape == (1, 3)
        assert (result == 1).all()

    def test_count_5_color_2(self):
        result = _render_count_colored(5, 2)
        assert result.shape == (1, 5)
        assert (result == 2).all()

    def test_count_1(self):
        result = _render_count_colored(1, 7)
        assert result.shape == (1, 1)
        assert result[0, 0] == 7

    def test_count_zero_returns_1x1(self):
        result = _render_count_colored(0, 3)
        assert result.shape == (1, 1)

    def test_negative_returns_1x1(self):
        result = _render_count_colored(-1, 3)
        assert result.shape == (1, 1)

    def test_dtype_is_int8(self):
        result = _render_count_colored(4, 2)
        assert result.dtype == np.int8


class TestTileN:
    """Unit tests for the NUMBER+OBJECTS bridge: tile_n."""

    def test_tile_3x(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        result = _tile_n(grid, 3)
        assert result.shape == (2, 6)
        assert (result[:, 0:2] == grid).all()
        assert (result[:, 2:4] == grid).all()
        assert (result[:, 4:6] == grid).all()

    def test_tile_1x_returns_copy(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        result = _tile_n(grid, 1)
        assert result.shape == (2, 2)
        assert (result == grid).all()

    def test_tile_0x_returns_grid(self):
        grid = np.array([[1, 2]], dtype=np.int8)
        result = _tile_n(grid, 0)
        assert result.shape == (1, 2)

    def test_tile_2x_row(self):
        grid = np.array([[3, 4, 5]], dtype=np.int8)
        result = _tile_n(grid, 2)
        assert result.shape == (1, 6)
        expected = np.array([[3, 4, 5, 3, 4, 5]], dtype=np.int8)
        assert (result == expected).all()


class TestRegistration:
    def test_registration_count(self, registry):
        assert len(registry) >= 25  # Should have at least 25 object primitives

    def test_all_tagged_objects(self, registry):
        from src.spelke_dsl.base import SpelkeSystem
        for prim in registry:
            assert prim.system == SpelkeSystem.OBJECTS

    def test_extract_callable(self, registry, sample_grid):
        extract = registry["extract_objects"]
        objs = extract(sample_grid)
        assert len(objs) == 2

    def test_render_count_colored_registered(self):
        # render_count_colored moved to l_number (SpelkeSystem.NUMBER) — check full library
        from src.spelke_dsl import build_spelke_library
        from src.spelke_dsl.base import SpelkeSystem
        reg = build_spelke_library()
        assert "render_count_colored" in reg._primitives
        assert reg["render_count_colored"].system == SpelkeSystem.NUMBER

    def test_tile_n_registered(self):
        # tile_n moved to l_number (SpelkeSystem.NUMBER) — check full library
        from src.spelke_dsl import build_spelke_library
        from src.spelke_dsl.base import SpelkeSystem
        reg = build_spelke_library()
        assert "tile_n" in reg._primitives
        assert reg["tile_n"].system == SpelkeSystem.NUMBER
