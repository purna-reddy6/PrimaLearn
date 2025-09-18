"""
l_places.py — PLACES Core Knowledge Module (Spelke System 4)

Implements Spelke's PLACES system: geometric layout, spatial frames,
reorientation, and landmark-based navigation.

Developmental evidence (Hermer & Spelke 1994, 1996; Lee & Spelke 2010):
Toddlers reorient using the geometric shape of the enclosing space
(distance + direction of surfaces) but initially ignore non-geometric
features (color, landmarks). By age ~6 they integrate landmarks with
geometry — a classic PrimaLearn case.

In the ARC domain, "places" manifest as:
  - Geometric layout of the grid (quadrants, regions, axes)
  - Distance fields from edges/corners/centers
  - Spatial frames (relative coordinates)
  - Region segmentation by geometric cues
  - Landmark-relative positioning

Design Note: These primitives encode SPATIAL STRUCTURE at the layout
level, complementing OBJECTS (what's in a location) and FORMS (how
shapes look). Cross-system abstractions like "object-at-landmark" or
"region-defined-by-geometry" bridge PLACES with OBJECTS.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tpoint, tlist, tpair,
)


# ──────────────────────────────────────────────────────────────────────
# Geometric Layout Primitives
# ──────────────────────────────────────────────────────────────────────

def _extract_quadrant(grid: np.ndarray, quadrant: int) -> np.ndarray:
    """
    Extract a quadrant from the grid.
    quadrant: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.
    """
    h, w = grid.shape
    mid_r, mid_c = h // 2, w // 2

    if quadrant == 0:
        return grid[:mid_r, :mid_c].copy()
    elif quadrant == 1:
        return grid[:mid_r, mid_c:].copy()
    elif quadrant == 2:
        return grid[mid_r:, :mid_c].copy()
    elif quadrant == 3:
        return grid[mid_r:, mid_c:].copy()
    else:
        return grid.copy()


def _get_center_region(grid: np.ndarray, margin: int = 1) -> np.ndarray:
    """Extract the center region excluding the border margin."""
    h, w = grid.shape
    if margin * 2 >= h or margin * 2 >= w:
        return grid.copy()
    return grid[margin:h-margin, margin:w-margin].copy()


def _get_border_region(grid: np.ndarray) -> np.ndarray:
    """Extract only the border cells (1-cell-wide frame)."""
    h, w = grid.shape
    result = np.zeros_like(grid)
    result[0, :] = grid[0, :]       # Top row
    result[h-1, :] = grid[h-1, :]   # Bottom row
    result[:, 0] = grid[:, 0]       # Left column
    result[:, w-1] = grid[:, w-1]   # Right column
    return result


def _split_horizontal(grid: np.ndarray) -> tuple:
    """Split grid into top and bottom halves."""
    h = grid.shape[0]
    mid = h // 2
    return grid[:mid].copy(), grid[mid:].copy()


def _split_vertical(grid: np.ndarray) -> tuple:
    """Split grid into left and right halves."""
    w = grid.shape[1]
    mid = w // 2
    return grid[:, :mid].copy(), grid[:, mid:].copy()


def _get_row(grid: np.ndarray, row: int) -> np.ndarray:
    """Extract a single row as a 1×W grid."""
    if 0 <= row < grid.shape[0]:
        return grid[row:row+1, :].copy()
    return np.zeros((1, grid.shape[1]), dtype=grid.dtype)


def _get_col(grid: np.ndarray, col: int) -> np.ndarray:
    """Extract a single column as an H×1 grid."""
    if 0 <= col < grid.shape[1]:
        return grid[:, col:col+1].copy()
    return np.zeros((grid.shape[0], 1), dtype=grid.dtype)


# ──────────────────────────────────────────────────────────────────────
# Distance / Spatial Field Primitives
# ──────────────────────────────────────────────────────────────────────

def _distance_from_edge(grid: np.ndarray) -> np.ndarray:
    """
    Compute the minimum distance from each cell to the nearest edge.
    Returns a grid where each cell's value = min(dist_to_border) mod 10.

    This is the geometric reorientation cue: layout encoded by
    distance from boundaries (Hermer & Spelke 1994).
    """
    h, w = grid.shape
    result = np.zeros((h, w), dtype=grid.dtype)
    for r in range(h):
        for c in range(w):
            d = min(r, c, h - 1 - r, w - 1 - c)
            result[r, c] = d % 10
    return result


def _distance_from_center(grid: np.ndarray) -> np.ndarray:
    """
    Compute the Manhattan distance from each cell to the grid center.
    Returns a grid where each cell's value = dist_to_center mod 10.
    """
    h, w = grid.shape
    cr, cc = h // 2, w // 2
    result = np.zeros((h, w), dtype=grid.dtype)
    for r in range(h):
        for c in range(w):
            d = abs(r - cr) + abs(c - cc)
            result[r, c] = d % 10
    return result


def _distance_from_color(grid: np.ndarray, target_color: int) -> np.ndarray:
    """
    Compute the Manhattan distance from each cell to the nearest cell
    of the target color. Returns distances mod 10. 0 if cell IS the target.

    This is landmark-relative positioning — distance from a distinguished
    feature of the environment.
    """
    from collections import deque

    h, w = grid.shape
    dist = np.full((h, w), 255, dtype=np.int32)
    queue = deque()

    # Seed with all cells of target_color
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == target_color:
                dist[r, c] = 0
                queue.append((r, c))

    # BFS
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    return (dist % 10).astype(grid.dtype)


# ──────────────────────────────────────────────────────────────────────
# Region Segmentation
# ──────────────────────────────────────────────────────────────────────

def _segment_by_axes(grid: np.ndarray) -> np.ndarray:
    """
    Segment the grid into regions by its central axes.
    Each quadrant gets a different color label (1-4).
    Pixels on the axes get color 0 (boundary).
    """
    h, w = grid.shape
    mid_r, mid_c = h // 2, w // 2
    result = np.zeros_like(grid)

    for r in range(h):
        for c in range(w):
            if r == mid_r or c == mid_c:
                result[r, c] = 0  # Axis
            elif r < mid_r and c < mid_c:
                result[r, c] = 1  # TL
            elif r < mid_r and c > mid_c:
                result[r, c] = 2  # TR
            elif r > mid_r and c < mid_c:
                result[r, c] = 3  # BL
            else:
                result[r, c] = 4  # BR

    return result


def _reflect_across_axis(grid: np.ndarray, axis: int) -> np.ndarray:
    """
    Reflect non-background content across the specified axis.
    axis: 0 = horizontal midline, 1 = vertical midline.

    This is the spatial reorientation primitive: using geometric
    properties of the enclosing space to determine locations.
    """
    h, w = grid.shape
    result = grid.copy()

    if axis == 0:  # Horizontal midline
        mid = h // 2
        for r in range(mid):
            for c in range(w):
                mirror_r = h - 1 - r
                if grid[r, c] != 0 and result[mirror_r, c] == 0:
                    result[mirror_r, c] = grid[r, c]
                elif grid[mirror_r, c] != 0 and result[r, c] == 0:
                    result[r, c] = grid[mirror_r, c]
    elif axis == 1:  # Vertical midline
        mid = w // 2
        for r in range(h):
            for c in range(mid):
                mirror_c = w - 1 - c
                if grid[r, c] != 0 and result[r, mirror_c] == 0:
                    result[r, mirror_c] = grid[r, c]
                elif grid[r, mirror_c] != 0 and result[r, c] == 0:
                    result[r, c] = grid[r, mirror_c]

    return result


def _reorient(grid: np.ndarray) -> np.ndarray:
    """
    Canonical reorientation: rotate grid so that the densest quadrant
    is in the top-left. This is the geometric reorientation behavior
    observed in toddlers navigating rectangular rooms.
    """
    h, w = grid.shape
    mid_r, mid_c = h // 2, w // 2

    densities = [
        np.count_nonzero(grid[:mid_r, :mid_c]),  # TL
        np.count_nonzero(grid[:mid_r, mid_c:]),   # TR
        np.count_nonzero(grid[mid_r:, :mid_c]),   # BL
        np.count_nonzero(grid[mid_r:, mid_c:]),   # BR
    ]

    densest = int(np.argmax(densities))

    if densest == 0:
        return grid.copy()  # Already TL
    elif densest == 1:
        return np.fliplr(grid).copy()  # Flip horizontally
    elif densest == 2:
        return np.flipud(grid).copy()  # Flip vertically
    else:
        return np.rot90(grid, k=2).copy()  # Rotate 180


# ──────────────────────────────────────────────────────────────────────
# Quadrant placement primitive (FORMS+OBJECTS+PLACES crossing)
# ──────────────────────────────────────────────────────────────────────

def _place_in_quadrant_8x8(obj_grid: np.ndarray, quadrant: int) -> np.ndarray:
    """
    FORMS+OBJECTS+PLACES cross-system primitive: place a small grid into the
    specified quadrant of a blank 8×8 canvas.

    Developmental grounding: Spelke's PLACES system encodes spatial frames
    (Hermer & Spelke 1994): "where in the room?" — top-right, bottom-left, etc.
    This primitive embodies landmark-relative placement: the object is anchored
    to a geometric region of the output space.

    Quadrant encoding (matching quad_tl / quad_tr / quad_bl / quad_br constants):
      0 = top-left     (rows 0-3, cols 0-3)
      1 = top-right    (rows 0-3, cols 4-7)
      2 = bottom-left  (rows 4-7, cols 0-3)
      3 = bottom-right (rows 4-7, cols 4-7)

    The obj_grid is placed at the top-left corner of the target quadrant region.
    If obj_grid is larger than 4×4, it is clipped to fit.

    Args:
        obj_grid: The object grid to place (should be ≤4×4).
        quadrant: Integer 0-3 selecting the target quadrant.

    Returns:
        np.ndarray of shape (8, 8) with obj_grid placed in the target quadrant.
    """
    canvas = np.zeros((8, 8), dtype=np.int8)

    # Quadrant top-left corner (in canvas coordinates)
    quad_origins = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 4)}
    r0, c0 = quad_origins.get(int(quadrant), (0, 0))

    oh, ow = obj_grid.shape
    # Clip to 4×4 if necessary
    max_h, max_w = 4, 4
    oh_clip = min(oh, max_h)
    ow_clip = min(ow, max_w)

    canvas[r0:r0 + oh_clip, c0:c0 + ow_clip] = obj_grid[:oh_clip, :ow_clip]
    return canvas


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_place_primitives(registry: PrimitiveRegistry) -> None:
    """Register all PLACES core knowledge primitives."""
    P = SpelkeSystem.PLACES

    primitives = [
        # ── Quadrant / Region extraction ──
        Primitive(
            "extract_quadrant", Arrow(tgrid, Arrow(tint, tgrid)),
            lambda g: lambda q: _extract_quadrant(g, q),
            P, "Extract quadrant (0=TL, 1=TR, 2=BL, 3=BR)",
            spelke_properties=["geometric-layout"],
        ),
        Primitive(
            "get_center", Arrow(tgrid, tgrid),
            lambda g: _get_center_region(g),
            P, "Extract center region excluding 1-cell border",
            spelke_properties=["geometric-layout"],
        ),
        Primitive(
            "get_border", Arrow(tgrid, tgrid),
            lambda g: _get_border_region(g),
            P, "Extract 1-cell-wide border frame",
            spelke_properties=["geometric-layout"],
        ),
        Primitive(
            "get_row", Arrow(tgrid, Arrow(tint, tgrid)),
            lambda g: lambda r: _get_row(g, r),
            P, "Extract a single row as a 1×W grid",
            spelke_properties=["spatial-frame"],
        ),
        Primitive(
            "get_col", Arrow(tgrid, Arrow(tint, tgrid)),
            lambda g: lambda c: _get_col(g, c),
            P, "Extract a single column as an H×1 grid",
            spelke_properties=["spatial-frame"],
        ),

        # ── Distance fields ──
        Primitive(
            "distance_from_edge", Arrow(tgrid, tgrid),
            _distance_from_edge,
            P, "Distance from nearest edge (mod 10) — geometric reorientation cue",
            spelke_properties=["geometric-layout", "reorientation"],
        ),
        Primitive(
            "distance_from_center", Arrow(tgrid, tgrid),
            _distance_from_center,
            P, "Manhattan distance from grid center (mod 10)",
            spelke_properties=["geometric-layout"],
        ),
        Primitive(
            "distance_from_color", Arrow(tgrid, Arrow(tcolor, tgrid)),
            lambda g: lambda c: _distance_from_color(g, c),
            P, "Distance from nearest cell of given color (mod 10) — landmark positioning",
            spelke_properties=["landmark", "spatial-frame"],
        ),

        # ── Spatial segmentation ──
        Primitive(
            "segment_by_axes", Arrow(tgrid, tgrid),
            _segment_by_axes,
            P, "Segment grid into 4 quadrants by central axes (labels 1-4)",
            spelke_properties=["geometric-layout"],
        ),
        Primitive(
            "reflect_across_h", Arrow(tgrid, tgrid),
            lambda g: _reflect_across_axis(g, 0),
            P, "Reflect content across horizontal midline",
            spelke_properties=["reorientation"],
        ),
        Primitive(
            "reflect_across_v", Arrow(tgrid, tgrid),
            lambda g: _reflect_across_axis(g, 1),
            P, "Reflect content across vertical midline",
            spelke_properties=["reorientation"],
        ),
        Primitive(
            "reorient", Arrow(tgrid, tgrid),
            _reorient,
            P, "Canonical reorientation: densest quadrant to top-left",
            spelke_properties=["reorientation", "geometric-layout"],
        ),

        # ── Quadrant constants ──
        Primitive("quad_tl", tint, 0, P, "Quadrant: top-left"),
        Primitive("quad_tr", tint, 1, P, "Quadrant: top-right"),
        Primitive("quad_bl", tint, 2, P, "Quadrant: bottom-left"),
        Primitive("quad_br", tint, 3, P, "Quadrant: bottom-right"),

        # ── FORMS+OBJECTS+PLACES cross-system bridge ──
        Primitive(
            "place_in_quadrant_8x8", Arrow(tgrid, Arrow(tint, tgrid)),
            lambda g: lambda q: _place_in_quadrant_8x8(g, q),
            P, "FORMS+OBJECTS+PLACES bridge: place obj_grid into quadrant of 8×8 canvas",
            spelke_properties=["geometric-layout", "spatial-frame"],
        ),
    ]

    for p in primitives:
        registry.register(p)
"""

PLACES module primitives: 16 total
- 5 region extraction (extract_quadrant, get_center, get_border, get_row, get_col)
- 3 distance fields (distance_from_edge, distance_from_center, distance_from_color)
- 4 spatial segmentation (segment_by_axes, reflect_across_h/v, reorient)
- 4 quadrant constants
"""
