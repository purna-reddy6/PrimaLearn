"""
strategies_v2.py — Advanced strategy families for ARC tasks.

Covers the remaining task families not handled by v1:
- Conditional per-object transforms
- Pattern continuation / extrapolation
- Grid layering / multi-object composition
- Pathfinding / connectivity
- Shape completion
- Spatial relation rules
- Color mapping by position
- Row/column operations
"""

from __future__ import annotations
import numpy as np
from typing import Callable
from collections import Counter, defaultdict
from src.spelke_dsl.l_objects import _extract_objects, GridObject


def verify(fn, pairs):
    for inp, expected in pairs:
        try:
            out = fn(inp)
            if out is None or not isinstance(out, np.ndarray):
                return False
            if out.shape != expected.shape or not np.array_equal(out, expected):
                return False
        except Exception:
            return False
    return True


def try_conditional_recolor(pairs):
    """Different recolor rules based on object properties."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    in_objs = _extract_objects(inp0)
    out_objs = _extract_objects(out0)
    if not in_objs or len(in_objs) != len(out_objs):
        return None
    
    # Match by position and build color→color map per old_color
    color_map = {}
    for io in in_objs:
        for oo in out_objs:
            if io.cells == oo.cells and io.color != oo.color:
                color_map[io.color] = oo.color
    
    if not color_map:
        return None
    
    def apply_conditional(g, cm=color_map):
        result = g.copy()
        objs = _extract_objects(g)
        for o in objs:
            if o.color in cm:
                for r, c in o.cells:
                    result[r, c] = cm[o.color]
        return result
    
    if verify(apply_conditional, pairs):
        return apply_conditional
    return None


def try_pattern_extrapolation(pairs):
    """Continue a pattern (e.g., growing sequence)."""
    if len(pairs) < 2:
        return None
    
    # Check if output extends input by a consistent pattern
    inp0, out0 = pairs[0]
    if inp0.shape == out0.shape:
        return None
    
    # Check: output is input with rows/cols added following a pattern
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    if ow == iw and oh > ih:
        # Rows added — check if new rows follow pattern from existing
        added = oh - ih
        existing = inp0
        new_rows = out0[ih:, :]
        
        # Try: last row repeated
        if added > 0:
            last_row = inp0[-1, :]
            predicted = np.tile(last_row, (added, 1))
            if np.array_equal(predicted, new_rows):
                def extend_last_row(g, n=added):
                    return np.vstack([g, np.tile(g[-1, :], (n, 1))])
                if verify(extend_last_row, pairs):
                    return extend_last_row
    
    return None


def try_color_by_neighbor_count(pairs):
    """Recolor cells based on number of non-zero neighbors."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    h, w = inp0.shape
    # Build neighbor count → output color mapping
    nc_map = {}
    consistent = True
    
    for r in range(h):
        for c in range(w):
            if inp0[r, c] == 0:
                continue
            count = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc_ = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc_ < w and inp0[nr, nc_] != 0:
                    count += 1
            oc = int(out0[r, c])
            if count in nc_map and nc_map[count] != oc:
                consistent = False
                break
            nc_map[count] = oc
        if not consistent:
            break
    
    if not consistent or not nc_map:
        return None
    
    def recolor_by_neighbors(g, nm=nc_map):
        h, w = g.shape
        result = g.copy()
        for r in range(h):
            for c in range(w):
                if g[r, c] == 0:
                    continue
                count = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc_ = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc_ < w and g[nr, nc_] != 0:
                        count += 1
                if count in nm:
                    result[r, c] = nm[count]
        return result
    
    if verify(recolor_by_neighbors, pairs):
        return recolor_by_neighbors
    return None


def try_fill_between_objects(pairs):
    """Fill the region between two objects."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    objs = _extract_objects(inp0)
    if len(objs) != 2:
        return None
    
    o1, o2 = objs
    fill_color = None
    
    diff = (inp0 != out0)
    if not diff.any():
        return None
    
    changed = list(zip(*np.where(diff)))
    if not changed:
        return None
    
    fill_color = int(out0[changed[0]])
    if not all(out0[r, c] == fill_color for r, c in changed):
        return None
    if not all(inp0[r, c] == 0 for r, c in changed):
        return None
    
    # Try: fill row between objects
    def fill_between_h(g, fc=fill_color):
        objects = _extract_objects(g)
        if len(objects) != 2:
            return g
        result = g.copy()
        o1, o2 = objects
        for r in range(g.shape[0]):
            cols1 = [c for rr, c in o1.cells if rr == r]
            cols2 = [c for rr, c in o2.cells if rr == r]
            if cols1 and cols2:
                cmin = min(max(cols1), max(cols2))
                cmax = max(min(cols1), min(cols2))
                if cmin < cmax:
                    for c in range(cmin+1, cmax):
                        if result[r, c] == 0:
                            result[r, c] = fc
        return result
    
    if verify(fill_between_h, pairs):
        return fill_between_h
    
    def fill_between_v(g, fc=fill_color):
        objects = _extract_objects(g)
        if len(objects) != 2:
            return g
        result = g.copy()
        o1, o2 = objects
        for c in range(g.shape[1]):
            rows1 = [r for r, cc in o1.cells if cc == c]
            rows2 = [r for r, cc in o2.cells if cc == c]
            if rows1 and rows2:
                rmin = min(max(rows1), max(rows2))
                rmax = max(min(rows1), min(rows2))
                if rmin < rmax:
                    for r in range(rmin+1, rmax):
                        if result[r, c] == 0:
                            result[r, c] = fc
        return result
    
    if verify(fill_between_v, pairs):
        return fill_between_v
    return None


def try_row_col_ops(pairs):
    """Operations on individual rows/columns."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    h, w = inp0.shape
    
    # Try: reverse each row
    def reverse_rows(g):
        return np.fliplr(g)
    if verify(reverse_rows, pairs):
        return reverse_rows
    
    # Try: reverse each column
    def reverse_cols(g):
        return np.flipud(g)
    if verify(reverse_cols, pairs):
        return reverse_cols
    
    # Try: shift rows cyclically
    for shift in range(1, w):
        def shift_rows(g, s=shift):
            return np.roll(g, s, axis=1)
        if verify(shift_rows, pairs):
            return shift_rows
    
    for shift in range(1, h):
        def shift_cols(g, s=shift):
            return np.roll(g, s, axis=0)
        if verify(shift_cols, pairs):
            return shift_cols
    
    return None


def try_inpaint_from_context(pairs):
    """Replace specific color cells using surrounding context."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    diff = (inp0 != out0)
    if not diff.any():
        return None
    
    # Find which input color gets replaced
    changed_positions = list(zip(*np.where(diff)))
    replaced_colors = set(int(inp0[r, c]) for r, c in changed_positions)
    
    if len(replaced_colors) != 1:
        return None
    
    marker_color = replaced_colors.pop()
    
    # Check if replacement comes from most common neighbor color
    def inpaint_majority(g, mc=marker_color):
        h, w = g.shape
        result = g.copy()
        for r in range(h):
            for c in range(w):
                if g[r, c] != mc:
                    continue
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != mc and g[nr, nc] != 0:
                        neighbors.append(int(g[nr, nc]))
                if neighbors:
                    result[r, c] = Counter(neighbors).most_common(1)[0][0]
        return result
    
    if verify(inpaint_majority, pairs):
        return inpaint_majority
    return None


def try_outline_objects(pairs):
    """Draw outlines around objects."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    objs = _extract_objects(inp0)
    if not objs:
        return None
    
    # Check if output adds border pixels around each object
    diff = (inp0 != out0)
    if not diff.any():
        return None
    
    outline_color = None
    changed = list(zip(*np.where(diff)))
    colors = set(int(out0[r, c]) for r, c in changed)
    if len(colors) != 1:
        return None
    outline_color = colors.pop()
    
    def add_outlines(g, oc=outline_color):
        h, w = g.shape
        result = g.copy()
        for r in range(h):
            for c in range(w):
                if g[r, c] != 0:
                    continue
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0:
                        result[r, c] = oc
                        break
        return result
    
    if verify(add_outlines, pairs):
        return add_outlines
    return None


def try_diagonal_fill(pairs):
    """Fill diagonals with colors."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    h, w = inp0.shape
    
    # Check if output has diagonal pattern based on input markers
    objs = _extract_objects(inp0)
    if not objs:
        return None
    
    # Try: extend each colored cell diagonally
    def extend_diag(g):
        h, w = g.shape
        result = g.copy()
        for r in range(h):
            for c in range(w):
                if g[r, c] == 0:
                    continue
                color = g[r, c]
                for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if result[nr, nc] == 0:
                            result[nr, nc] = color
                        nr += dr
                        nc += dc
        return result
    
    if verify(extend_diag, pairs):
        return extend_diag
    
    # Try: extend horizontally and vertically (cross pattern)
    def extend_cross(g):
        h, w = g.shape
        result = g.copy()
        for r in range(h):
            for c in range(w):
                if g[r, c] == 0:
                    continue
                color = g[r, c]
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if result[nr, nc] == 0:
                            result[nr, nc] = color
                        else:
                            break
                        nr += dr
                        nc += dc
        return result
    
    if verify(extend_cross, pairs):
        return extend_cross
    return None


def try_color_map_by_position(pairs):
    """Map colors based on object position (left→right, top→bottom)."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    objs = _extract_objects(inp0)
    if len(objs) < 2:
        return None
    
    out_objs = _extract_objects(out0)
    if len(objs) != len(out_objs):
        return None
    
    # Sort by position, check if colors cycle or follow position order
    sorted_in = sorted(objs, key=lambda o: (o.top_left[1], o.top_left[0]))
    sorted_out = sorted(out_objs, key=lambda o: (o.top_left[1], o.top_left[0]))
    
    # Position-based color assignment
    pos_to_color = {}
    for i, (io, oo) in enumerate(zip(sorted_in, sorted_out)):
        if io.cells != oo.cells:
            return None
        pos_to_color[i] = oo.color
    
    if len(set(pos_to_color.values())) == 1:
        return None  # All same color, not interesting
    
    def recolor_by_position(g, ptc=pos_to_color):
        objects = _extract_objects(g)
        sorted_objs = sorted(objects, key=lambda o: (o.top_left[1], o.top_left[0]))
        result = g.copy()
        for i, o in enumerate(sorted_objs):
            if i in ptc:
                for r, c in o.cells:
                    result[r, c] = ptc[i]
        return result
    
    if verify(recolor_by_position, pairs):
        return recolor_by_position
    return None


def try_max_overlap(pairs):
    """Keep the color that appears at each position across multiple overlaid objects."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    objs = _extract_objects(inp0)
    if len(objs) < 2:
        return None
    
    # Try: at each cell, keep the non-zero value if unique, or resolve by majority
    def majority_overlap(g):
        h, w = g.shape
        objects = _extract_objects(g)
        result = np.zeros_like(g)
        cell_colors = defaultdict(list)
        for o in objects:
            for r, c in o.cells:
                cell_colors[(r, c)].append(o.color)
        for (r, c), colors in cell_colors.items():
            result[r, c] = Counter(colors).most_common(1)[0][0]
        # Keep non-object cells
        for r in range(h):
            for c in range(w):
                if (r, c) not in cell_colors:
                    result[r, c] = g[r, c]
        return result
    
    if verify(majority_overlap, pairs):
        return majority_overlap
    return None


def try_extract_repeating(pairs):
    """Extract the repeating unit from a periodic grid."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    if oh >= ih or ow >= iw:
        return None
    
    # Check if input is tiling of output
    if ih % oh == 0 and iw % ow == 0:
        tiled = np.tile(out0, (ih // oh, iw // ow))
        if tiled.shape == inp0.shape and np.array_equal(tiled, inp0):
            def extract_tile(g):
                h, w = g.shape
                for ph in range(1, h):
                    if h % ph != 0:
                        continue
                    for pw in range(1, w):
                        if w % pw != 0:
                            continue
                        pat = g[:ph, :pw]
                        if np.array_equal(np.tile(pat, (h//ph, w//pw)), g):
                            return pat.copy()
                return g
            if verify(extract_tile, pairs):
                return extract_tile
    return None


ALL_EXTENDED_STRATEGIES_V2 = [
    ("conditional_recolor", try_conditional_recolor),
    ("pattern_extrapolation", try_pattern_extrapolation),
    ("color_by_neighbor", try_color_by_neighbor_count),
    ("fill_between", try_fill_between_objects),
    ("row_col_ops", try_row_col_ops),
    ("inpaint", try_inpaint_from_context),
    ("outline_objects", try_outline_objects),
    ("diagonal_fill", try_diagonal_fill),
    ("color_by_position", try_color_map_by_position),
    ("max_overlap", try_max_overlap),
    ("extract_repeating", try_extract_repeating),
]
