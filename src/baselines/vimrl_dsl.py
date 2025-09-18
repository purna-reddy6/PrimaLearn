"""
vimrl_dsl.py — VIMRL-style objectness-only DSL baseline.

Based on Ainooson et al. (2023) "An Approach for Solving Tasks on the
ARC-AGI Benchmark" — an objectness-focused DSL that uses only spatial
and perceptual object primitives without the broader Spelke library.

This is Comparison Condition 5 from the master plan (Section 11.3):
  "VIMRL baseline (Ainooson et al. 2023) for an objectness-only DSL comparison"

Purpose: isolates the contribution of multi-module Spelke initialization
from objectness priors alone. If Spelke > VIMRL > Generic, that shows
each additional module adds measurable signal.

The VIMRL DSL covers:
  - Object extraction and representation
  - Spatial relationships (above, below, adjacent, inside)
  - Object properties (color, size, shape, position)
  - Simple transformations (move, resize, recolor)
  - Grid operations on objects

What it deliberately LACKS (vs full Spelke):
  - L_forms: symmetry, rotation group, geometric regularity
  - L_number: counting, cardinality, arithmetic
  - L_analogy: cross-domain structural alignment
  - L_agents / L_persons / L_places modules
"""

from __future__ import annotations
import numpy as np
from typing import Any
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tpoint, tlist, tpair,
    t0, t1,
)


def _safe(fn, default=None):
    def wrapped(*args):
        try:
            return fn(*args)
        except Exception:
            return default
    return wrapped


def build_vimrl_dsl() -> PrimitiveRegistry:
    """
    Build the VIMRL objectness-only DSL.

    Covers object extraction, spatial relations, and simple transformations.
    No geometry module, no number module, no analogy.
    """
    registry = PrimitiveRegistry()
    O = SpelkeSystem.OBJECTS
    G = SpelkeSystem.GLUE

    # ── Object extraction ──────────────────────────────────────────────
    def _extract_objects(grid):
        """4-connected component extraction — core VIMRL operation."""
        if not hasattr(grid, '__len__'):
            return []
        try:
            g = np.array(grid) if not isinstance(grid, np.ndarray) else grid
            if g.ndim != 2:
                return []
            bg = 0
            visited = np.zeros_like(g, dtype=bool)
            objects = []
            for r in range(g.shape[0]):
                for c in range(g.shape[1]):
                    if not visited[r, c] and g[r, c] != bg:
                        # BFS
                        color = g[r, c]
                        cells = []
                        queue = [(r, c)]
                        visited[r, c] = True
                        while queue:
                            cr, cc = queue.pop(0)
                            cells.append((cr, cc))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                                    if not visited[nr, nc] and g[nr, nc] == color:
                                        visited[nr, nc] = True
                                        queue.append((nr, nc))
                        objects.append({"color": int(color), "cells": cells,
                                        "size": len(cells),
                                        "bbox": (min(r for r,_ in cells),
                                                 min(c for _,c in cells),
                                                 max(r for r,_ in cells),
                                                 max(c for _,c in cells))})
            return objects
        except Exception:
            return []

    def _object_color(obj):
        return obj.get("color", 0) if isinstance(obj, dict) else 0

    def _object_size(obj):
        return obj.get("size", 0) if isinstance(obj, dict) else 0

    def _object_position(obj):
        if isinstance(obj, dict) and "bbox" in obj:
            r0, c0, r1, c1 = obj["bbox"]
            return (r0, c0)
        return (0, 0)

    def _objects_adjacent(o1, o2):
        if not (isinstance(o1, dict) and isinstance(o2, dict)):
            return False
        cells1 = set(map(tuple, o1.get("cells", [])))
        for r, c in o2.get("cells", []):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (r+dr, c+dc) in cells1:
                    return True
        return False

    def _objects_same_color(o1, o2):
        return _object_color(o1) == _object_color(o2)

    def _objects_same_size(o1, o2):
        return _object_size(o1) == _object_size(o2)

    def _largest_object(objects):
        if not objects:
            return None
        return max(objects, key=lambda o: o.get("size", 0))

    def _smallest_object(objects):
        if not objects:
            return None
        return min(objects, key=lambda o: o.get("size", 0))

    def _filter_by_color(color):
        return lambda objects: [o for o in objects if o.get("color") == color]

    def _sort_by_size(objects):
        return sorted(objects, key=lambda o: o.get("size", 0), reverse=True)

    def _objects_above(o1, o2):
        if not (isinstance(o1, dict) and isinstance(o2, dict)):
            return False
        r1 = o1.get("bbox", (0,0,0,0))[0]
        r2 = o2.get("bbox", (0,0,0,0))[0]
        return r1 < r2

    def _paint_object(grid, obj, new_color):
        try:
            g = [list(row) for row in grid]
            for r, c in obj.get("cells", []):
                if 0 <= r < len(g) and 0 <= c < len(g[0]):
                    g[r][c] = new_color
            return [tuple(row) for row in g]
        except Exception:
            return grid

    def _move_object(grid, obj, dr, dc):
        try:
            g = np.array(grid)
            result = np.zeros_like(g)
            for r, c in obj.get("cells", []):
                nr, nc = r + dr, c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    result[nr, nc] = g[r, c]
                else:
                    result[r, c] = g[r, c]
            return result.tolist()
        except Exception:
            return grid

    def _background_color(grid):
        try:
            g = np.array(grid).flatten()
            vals, counts = np.unique(g, return_counts=True)
            return int(vals[np.argmax(counts)])
        except Exception:
            return 0

    def _grid_width(grid):
        try:
            return len(grid[0]) if grid else 0
        except Exception:
            return 0

    def _grid_height(grid):
        try:
            return len(grid)
        except Exception:
            return 0

    def _objects_touching_border(grid, obj):
        try:
            h, w = len(grid), len(grid[0])
            for r, c in obj.get("cells", []):
                if r == 0 or r == h-1 or c == 0 or c == w-1:
                    return True
            return False
        except Exception:
            return False

    def _fill_object_bbox(grid, obj, color):
        try:
            g = [list(row) for row in grid]
            r0, c0, r1, c1 = obj.get("bbox", (0,0,0,0))
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    if 0 <= r < len(g) and 0 <= c < len(g[0]):
                        g[r][c] = color
            return [tuple(row) for row in g]
        except Exception:
            return grid

    def _copy_object_to(grid, obj, target_r, target_c):
        try:
            g = [list(row) for row in grid]
            r0 = min(r for r, _ in obj.get("cells", [(0,0)]))
            c0 = min(c for _, c in obj.get("cells", [(0,0)]))
            for r, c in obj.get("cells", []):
                nr = r - r0 + target_r
                nc = c - c0 + target_c
                if 0 <= nr < len(g) and 0 <= nc < len(g[0]):
                    g[nr][nc] = obj.get("color", 0)
            return [tuple(row) for row in g]
        except Exception:
            return grid

    def _empty_grid(h, w, color=0):
        return [[color]*w for _ in range(h)]

    # ── Register all primitives ────────────────────────────────────────

    prims = [
        # Core extraction
        Primitive("v_extract_objects", Arrow(tgrid, tlist(tobject)),
                  _extract_objects, O, "extract 4-connected objects"),
        Primitive("v_background_color", Arrow(tgrid, tcolor),
                  _background_color, O, "most frequent color (background)"),
        Primitive("v_grid_width", Arrow(tgrid, tint),
                  _grid_width, O, "grid width"),
        Primitive("v_grid_height", Arrow(tgrid, tint),
                  _grid_height, O, "grid height"),

        # Object properties
        Primitive("v_obj_color", Arrow(tobject, tcolor),
                  _object_color, O, "object color"),
        Primitive("v_obj_size", Arrow(tobject, tint),
                  _object_size, O, "object pixel count"),
        Primitive("v_obj_position", Arrow(tobject, tpoint),
                  _object_position, O, "object top-left position"),

        # Object selection
        Primitive("v_largest", Arrow(tlist(tobject), tobject),
                  _largest_object, O, "largest object by pixel count"),
        Primitive("v_smallest", Arrow(tlist(tobject), tobject),
                  _smallest_object, O, "smallest object by pixel count"),
        Primitive("v_sort_by_size", Arrow(tlist(tobject), tlist(tobject)),
                  _sort_by_size, O, "sort objects by size descending"),
        Primitive("v_filter_color", Arrow(tcolor, Arrow(tlist(tobject), tlist(tobject))),
                  _filter_by_color, O, "filter objects by color"),

        # Spatial relations
        Primitive("v_adjacent", Arrow(tobject, Arrow(tobject, tbool)),
                  lambda o1: lambda o2: _objects_adjacent(o1, o2), O,
                  "objects share a border cell"),
        Primitive("v_above", Arrow(tobject, Arrow(tobject, tbool)),
                  lambda o1: lambda o2: _objects_above(o1, o2), O,
                  "object 1 is above object 2"),
        Primitive("v_same_color", Arrow(tobject, Arrow(tobject, tbool)),
                  lambda o1: lambda o2: _objects_same_color(o1, o2), O,
                  "objects have same color"),
        Primitive("v_same_size", Arrow(tobject, Arrow(tobject, tbool)),
                  lambda o1: lambda o2: _objects_same_size(o1, o2), O,
                  "objects have same size"),
        Primitive("v_touching_border", Arrow(tgrid, Arrow(tobject, tbool)),
                  lambda g: lambda o: _objects_touching_border(g, o), O,
                  "object touches grid border"),

        # Transformations (grid-level)
        Primitive("v_paint_object", Arrow(tgrid, Arrow(tobject, Arrow(tcolor, tgrid))),
                  lambda g: lambda o: lambda c: _paint_object(g, o, c), O,
                  "recolor all cells of an object"),
        Primitive("v_move_object", Arrow(tgrid, Arrow(tobject, Arrow(tint, Arrow(tint, tgrid)))),
                  lambda g: lambda o: lambda dr: lambda dc: _move_object(g, o, dr, dc), O,
                  "translate object by (dr, dc)"),
        Primitive("v_fill_bbox", Arrow(tgrid, Arrow(tobject, Arrow(tcolor, tgrid))),
                  lambda g: lambda o: lambda c: _fill_object_bbox(g, o, c), O,
                  "fill bounding box of object with color"),
        Primitive("v_copy_to", Arrow(tgrid, Arrow(tobject, Arrow(tint, Arrow(tint, tgrid)))),
                  lambda g: lambda o: lambda r: lambda c: _copy_object_to(g, o, r, c), O,
                  "copy object to target position"),
        Primitive("v_empty_grid", Arrow(tint, Arrow(tint, tgrid)),
                  lambda h: lambda w: _empty_grid(h, w), O,
                  "create blank grid of given size"),

        # Color constants
        Primitive("v_color_0", tcolor, 0, O, "color: black"),
        Primitive("v_color_1", tcolor, 1, O, "color: blue"),
        Primitive("v_color_2", tcolor, 2, O, "color: red"),
        Primitive("v_color_3", tcolor, 3, O, "color: green"),
        Primitive("v_color_4", tcolor, 4, O, "color: yellow"),
        Primitive("v_color_5", tcolor, 5, O, "color: grey"),
        Primitive("v_color_6", tcolor, 6, O, "color: fuschia"),
        Primitive("v_color_7", tcolor, 7, O, "color: orange"),
        Primitive("v_color_8", tcolor, 8, O, "color: teal"),
        Primitive("v_color_9", tcolor, 9, O, "color: maroon"),

        # Small integers
        Primitive("v_int_0", tint, 0, O, "0"),
        Primitive("v_int_1", tint, 1, O, "1"),
        Primitive("v_int_neg1", tint, -1, O, "-1"),
        Primitive("v_int_2", tint, 2, O, "2"),

        # Generic combinators (minimal — VIMRL only needs these for composition)
        Primitive("v_compose", Arrow(Arrow(t1, t1), Arrow(Arrow(t0, t1), Arrow(t0, t1))),
                  lambda f: lambda g: lambda x: f(g(x)), G, "function composition"),
        Primitive("v_identity", Arrow(t0, t0),
                  lambda x: x, G, "identity"),
        Primitive("v_map", Arrow(Arrow(t0, t1), Arrow(tlist(t0), tlist(t1))),
                  lambda f: lambda xs: [f(x) for x in xs], G, "map"),
        Primitive("v_filter", Arrow(Arrow(t0, tbool), Arrow(tlist(t0), tlist(t0))),
                  lambda p: lambda xs: [x for x in xs if p(x)], G, "filter"),
        Primitive("v_first", Arrow(tlist(t0), t0),
                  lambda xs: xs[0] if xs else None, G, "first element"),
        Primitive("v_last", Arrow(tlist(t0), t0),
                  lambda xs: xs[-1] if xs else None, G, "last element"),
        Primitive("v_length", Arrow(tlist(t0), tint),
                  lambda xs: len(xs), G, "list length"),
        Primitive("v_if", Arrow(tbool, Arrow(t0, Arrow(t0, t0))),
                  lambda b: lambda t: lambda f: t if b else f, G, "if-then-else"),
        Primitive("v_eq", Arrow(t0, Arrow(t0, tbool)),
                  lambda a: lambda b: a == b, G, "equality"),
        Primitive("v_gt", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: a > b, G, "greater than"),
    ]

    for p in prims:
        registry.register(p)

    return registry
