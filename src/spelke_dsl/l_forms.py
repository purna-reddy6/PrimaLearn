"""
l_forms.py — Core System 5: Forms / Scale-Invariant Shape Geometry

Implements geometric primitives from Spelke's forms system.
Evidence: Munduruku studies (Dehaene, Izard, Pica & Spelke 2006, Science);
Sablé-Meyer et al. 2022 PNAS (geometric regularity); Atzeni et al. 2023 lattice symmetry.

Primitives: translation, rotation, reflection, repetition, symmetry detection,
line detection, scaling, group operations.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tpoint, tdir, tlist, tpair, taxis,
)


# ──────────────────────────────────────────────────────────────────────
# Grid-level geometric transformations
# ──────────────────────────────────────────────────────────────────────

def _rotate_grid_90(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90° clockwise."""
    return np.rot90(grid, k=-1).copy()

def _rotate_grid_180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=2).copy()

def _rotate_grid_270(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=-3).copy()

def _flip_horizontal(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid).copy()

def _flip_vertical(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid).copy()

def _transpose(grid: np.ndarray) -> np.ndarray:
    return grid.T.copy()


# ──────────────────────────────────────────────────────────────────────
# Symmetry detection
# ──────────────────────────────────────────────────────────────────────

def _rotate_n(grid: np.ndarray, n: int) -> np.ndarray:
    """
    BRIDGE PRIMITIVE (FORMS+NUMBER crossing): Rotate grid by n*90 degrees clockwise.

    n=0 → identity, n=1 → 90° CW, n=2 → 180°, n=3 → 270° CW.
    Handles any integer n via mod 4. Used in Type 3 and Type 6 tasks where
    the NUMBER system produces a count that drives a FORMS rotation.
    """
    k = int(n) % 4
    if k == 0:
        return grid.copy()
    elif k == 1:
        return np.rot90(grid, k=-1).copy()
    elif k == 2:
        return np.rot90(grid, k=2).copy()
    else:
        return np.rot90(grid, k=-3).copy()


def _has_horizontal_symmetry(grid: np.ndarray) -> bool:
    """Does grid have left-right mirror symmetry?"""
    return np.array_equal(grid, np.fliplr(grid))

def _has_vertical_symmetry(grid: np.ndarray) -> bool:
    """Does grid have top-bottom mirror symmetry?"""
    return np.array_equal(grid, np.flipud(grid))

def _has_rotational_symmetry_90(grid: np.ndarray) -> bool:
    """Does grid have 90° rotational symmetry?"""
    if grid.shape[0] != grid.shape[1]:
        return False
    return np.array_equal(grid, np.rot90(grid, k=-1))

def _has_rotational_symmetry_180(grid: np.ndarray) -> bool:
    """Does grid have 180° rotational symmetry?"""
    return np.array_equal(grid, np.rot90(grid, k=2))

def _has_diagonal_symmetry(grid: np.ndarray) -> bool:
    """Symmetry along main diagonal."""
    if grid.shape[0] != grid.shape[1]:
        return False
    return np.array_equal(grid, grid.T)

def _symmetry_axes(grid: np.ndarray) -> list[str]:
    """Detect all symmetry axes of a grid."""
    axes = []
    if _has_horizontal_symmetry(grid):
        axes.append("horizontal")
    if _has_vertical_symmetry(grid):
        axes.append("vertical")
    if _has_rotational_symmetry_180(grid):
        axes.append("rot180")
    if grid.shape[0] == grid.shape[1]:
        if _has_rotational_symmetry_90(grid):
            axes.append("rot90")
        if _has_diagonal_symmetry(grid):
            axes.append("diagonal")
        if np.array_equal(grid, np.fliplr(grid.T)):
            axes.append("anti_diagonal")
    return axes


# ──────────────────────────────────────────────────────────────────────
# Tiling and repetition
# ──────────────────────────────────────────────────────────────────────

def _tile_grid(pattern: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Tile a pattern n_rows × n_cols times."""
    return np.tile(pattern, (n_rows, n_cols))

def _is_periodic(grid: np.ndarray) -> bool:
    """Check if grid is a periodic tiling of a smaller pattern."""
    h, w = grid.shape
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0:
                continue
            if ph == h and pw == w:
                continue
            pattern = grid[:ph, :pw]
            if np.array_equal(np.tile(pattern, (h // ph, w // pw)), grid):
                return True
    return False

def _find_period(grid: np.ndarray) -> tuple[int, int]:
    """Find the minimal period of a periodic grid."""
    h, w = grid.shape
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0:
                continue
            if ph == h and pw == w:
                return (h, w)
            pattern = grid[:ph, :pw]
            if np.array_equal(np.tile(pattern, (h // ph, w // pw)), grid):
                return (ph, pw)
    return (h, w)

def _extract_repeating_unit(grid: np.ndarray) -> np.ndarray:
    """Extract the minimal repeating tile from a periodic grid."""
    ph, pw = _find_period(grid)
    return grid[:ph, :pw].copy()


# ──────────────────────────────────────────────────────────────────────
# Line detection and drawing
# ──────────────────────────────────────────────────────────────────────

def _draw_line(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int,
               color: int) -> np.ndarray:
    """Draw a line using Bresenham's algorithm."""
    g = grid.copy()
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            g[r, c] = color
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return g

def _detect_horizontal_lines(grid: np.ndarray, color: int) -> list[tuple[int, int, int]]:
    """Detect horizontal lines of a given color. Returns (row, col_start, col_end)."""
    lines = []
    h, w = grid.shape
    for r in range(h):
        start = None
        for c in range(w):
            if grid[r, c] == color:
                if start is None:
                    start = c
            else:
                if start is not None and c - start >= 2:
                    lines.append((r, start, c - 1))
                start = None
        if start is not None and w - start >= 2:
            lines.append((r, start, w - 1))
    return lines

def _detect_vertical_lines(grid: np.ndarray, color: int) -> list[tuple[int, int, int]]:
    """Detect vertical lines of a given color. Returns (col, row_start, row_end)."""
    lines = []
    h, w = grid.shape
    for c in range(w):
        start = None
        for r in range(h):
            if grid[r, c] == color:
                if start is None:
                    start = r
            else:
                if start is not None and r - start >= 2:
                    lines.append((c, start, r - 1))
                start = None
        if start is not None and h - start >= 2:
            lines.append((c, start, h - 1))
    return lines


# ──────────────────────────────────────────────────────────────────────
# Scaling
# ──────────────────────────────────────────────────────────────────────

def _scale_up(grid: np.ndarray, factor: int) -> np.ndarray:
    """Scale grid up by integer factor (nearest-neighbor)."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)

def _scale_down(grid: np.ndarray, factor: int) -> np.ndarray:
    """Scale grid down by integer factor (sample top-left of each block)."""
    if factor <= 0:
        return grid.copy()
    h, w = grid.shape
    nh, nw = h // factor, w // factor
    if nh == 0 or nw == 0:
        return grid.copy()
    return grid[:nh * factor:factor, :nw * factor:factor].copy()

def _resize_grid(grid: np.ndarray, new_h: int, new_w: int,
                 fill: int = 0) -> np.ndarray:
    """Resize grid to new dimensions, padding or cropping."""
    result = np.full((new_h, new_w), fill, dtype=grid.dtype)
    copy_h = min(grid.shape[0], new_h)
    copy_w = min(grid.shape[1], new_w)
    result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
    return result


# ──────────────────────────────────────────────────────────────────────
# Pattern matching and shape analysis
# ──────────────────────────────────────────────────────────────────────

def _grids_equal(g1: np.ndarray, g2: np.ndarray) -> bool:
    if g1.shape != g2.shape:
        return False
    return np.array_equal(g1, g2)

def _grid_diff(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Cells that differ between two same-sized grids (marked as 1)."""
    if g1.shape != g2.shape:
        raise ValueError("Grids must be same size for diff")
    return (g1 != g2).astype(np.int8)

def _count_colors(grid: np.ndarray) -> int:
    return len(set(int(x) for x in grid.flat))

def _color_histogram(grid: np.ndarray) -> dict[int, int]:
    from collections import Counter
    return dict(Counter(int(x) for x in grid.flat))

def _replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
    g = grid.copy()
    g[g == old_color] = new_color
    return g

def _grid_shape(grid: np.ndarray) -> tuple[int, int]:
    return grid.shape

def _is_square(grid: np.ndarray) -> bool:
    return grid.shape[0] == grid.shape[1]

def _make_border(grid: np.ndarray, color: int, width: int = 1) -> np.ndarray:
    """Add a border of given color and width around the grid."""
    h, w = grid.shape
    result = np.full((h + 2 * width, w + 2 * width), color, dtype=grid.dtype)
    result[width:width + h, width:width + w] = grid
    return result

def _hollow_rectangle(h: int, w: int, border_color: int,
                      fill_color: int = 0) -> np.ndarray:
    """Create a hollow rectangle."""
    g = np.full((h, w), fill_color, dtype=np.int8)
    g[0, :] = border_color
    g[-1, :] = border_color
    g[:, 0] = border_color
    g[:, -1] = border_color
    return g

def _overlay_grids(bottom: np.ndarray, top: np.ndarray,
                   transparent: int = 0) -> np.ndarray:
    """Overlay top grid on bottom, with transparent color pass-through."""
    if bottom.shape != top.shape:
        raise ValueError("Grids must be same size for overlay")
    result = bottom.copy()
    mask = top != transparent
    result[mask] = top[mask]
    return result


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_form_primitives(registry: PrimitiveRegistry) -> None:
    """Register all L_forms primitives."""
    S = SpelkeSystem.FORMS
    props = ["domain-specific", "task-specific", "abstract"]

    primitives = [
        # Grid transformations
        Primitive("rotate_n", Arrow(tgrid, Arrow(tint, tgrid)),
                  lambda g: lambda n: _rotate_n(g, n), S,
                  "FORMS+NUMBER bridge: rotate grid by n*90 degrees clockwise (n=0,1,2,3)", props),
        Primitive("rotate90", Arrow(tgrid, tgrid), _rotate_grid_90, S,
                  "Rotate grid 90° clockwise", props),
        Primitive("rotate180", Arrow(tgrid, tgrid), _rotate_grid_180, S,
                  "Rotate grid 180°", props),
        Primitive("rotate270", Arrow(tgrid, tgrid), _rotate_grid_270, S,
                  "Rotate grid 270° clockwise", props),
        Primitive("flip_h", Arrow(tgrid, tgrid), _flip_horizontal, S,
                  "Flip grid horizontally", props),
        Primitive("flip_v", Arrow(tgrid, tgrid), _flip_vertical, S,
                  "Flip grid vertically", props),
        Primitive("transpose", Arrow(tgrid, tgrid), _transpose, S,
                  "Transpose grid", props),

        # Symmetry detection
        Primitive("sym_horizontal", Arrow(tgrid, tbool), _has_horizontal_symmetry, S,
                  "Test left-right mirror symmetry", props),
        Primitive("sym_vertical", Arrow(tgrid, tbool), _has_vertical_symmetry, S,
                  "Test top-bottom mirror symmetry", props),
        Primitive("sym_rot90", Arrow(tgrid, tbool), _has_rotational_symmetry_90, S,
                  "Test 90° rotational symmetry", props),
        Primitive("sym_rot180", Arrow(tgrid, tbool), _has_rotational_symmetry_180, S,
                  "Test 180° rotational symmetry", props),
        Primitive("sym_diagonal", Arrow(tgrid, tbool), _has_diagonal_symmetry, S,
                  "Test main-diagonal symmetry", props),

        # Tiling / repetition
        Primitive("tile", Arrow(tgrid, Arrow(tint, Arrow(tint, tgrid))),
                  lambda g: lambda nr: lambda nc: _tile_grid(g, nr, nc), S,
                  "Tile pattern n_rows × n_cols", props),
        Primitive("is_periodic", Arrow(tgrid, tbool), _is_periodic, S,
                  "Check if grid is a periodic tiling", props),
        Primitive("find_period", Arrow(tgrid, tpair(tint, tint)), _find_period, S,
                  "Find minimal period (height, width)", props),
        Primitive("extract_tile", Arrow(tgrid, tgrid), _extract_repeating_unit, S,
                  "Extract minimal repeating unit", props),

        # Lines
        Primitive("draw_line",
                  Arrow(tgrid, Arrow(tint, Arrow(tint, Arrow(tint, Arrow(tint, Arrow(tcolor, tgrid)))))),
                  lambda g: lambda r0: lambda c0: lambda r1: lambda c1: lambda col: _draw_line(g, r0, c0, r1, c1, col), S,
                  "Draw line (Bresenham)", props),

        # Scaling
        Primitive("scale_up", Arrow(tgrid, Arrow(tint, tgrid)),
                  lambda g: lambda f: _scale_up(g, f), S,
                  "Scale grid up by integer factor", props),
        Primitive("scale_down", Arrow(tgrid, Arrow(tint, tgrid)),
                  lambda g: lambda f: _scale_down(g, f), S,
                  "Scale grid down by integer factor", props),
        Primitive("resize", Arrow(tgrid, Arrow(tint, Arrow(tint, tgrid))),
                  lambda g: lambda h: lambda w: _resize_grid(g, h, w), S,
                  "Resize grid to h×w", props),

        # Comparison / analysis
        Primitive("grids_equal", Arrow(tgrid, Arrow(tgrid, tbool)),
                  _grids_equal, S, "Are two grids identical?", props),
        Primitive("grid_diff", Arrow(tgrid, Arrow(tgrid, tgrid)),
                  _grid_diff, S, "Compute cell-wise difference mask", props),
        Primitive("count_colors", Arrow(tgrid, tint), _count_colors, S,
                  "Count unique colors", props),
        Primitive("replace_color", Arrow(tgrid, Arrow(tcolor, Arrow(tcolor, tgrid))),
                  lambda g: lambda old: lambda new: _replace_color(g, old, new), S,
                  "Replace all cells of one color with another", props),
        Primitive("grid_height", Arrow(tgrid, tint), lambda g: g.shape[0], S,
                  "Grid height", props),
        Primitive("grid_width", Arrow(tgrid, tint), lambda g: g.shape[1], S,
                  "Grid width", props),
        Primitive("is_square", Arrow(tgrid, tbool), _is_square, S,
                  "Is grid square?", props),

        # Construction
        Primitive("make_border", Arrow(tgrid, Arrow(tcolor, tgrid)),
                  lambda g: lambda c: _make_border(g, c), S,
                  "Add 1-cell border", props),
        Primitive("hollow_rect", Arrow(tint, Arrow(tint, Arrow(tcolor, tgrid))),
                  lambda h: lambda w: lambda c: _hollow_rectangle(h, w, c), S,
                  "Create hollow rectangle", props),
        Primitive("overlay", Arrow(tgrid, Arrow(tgrid, tgrid)),
                  lambda b: lambda t: _overlay_grids(b, t), S,
                  "Overlay grids (0 = transparent)", props),
        Primitive("vstack_grid", Arrow(tgrid, Arrow(tgrid, tgrid)),
                  lambda a: lambda b: np.vstack([a, b]), S,
                  "Stack two grids vertically", props),
        Primitive("hstack_grid", Arrow(tgrid, Arrow(tgrid, tgrid)),
                  lambda a: lambda b: np.hstack([a, b]), S,
                  "Stack two grids horizontally", props),
    ]

    for p in primitives:
        registry.register(p)
