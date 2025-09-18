"""
strategies_v3.py — Advanced ARC strategy families (batch 3).

Targets patterns identified from ARC task analysis:
- Path tracing / connected component operations  
- Color frequency histograms
- Symmetry axis operations
- Template matching with wildcards
- Grid subdivision and recombination
- Projection / shadow casting
- Majority vote per region
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


def try_connected_component_ops(pairs):
    """Operations on connected components: keep/remove by property."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    objs = _extract_objects(inp0)
    if len(objs) < 2:
        return None
    
    out_nonzero = set(zip(*np.where(out0 != 0)))
    
    # Try: keep objects that are "enclosed" (surrounded by other objects)
    for obj in objs:
        obj_cells = set(obj.cells)
        # Check if this single object matches output
        if obj_cells == out_nonzero:
            # Find the distinguishing property
            if obj.size == max(o.size for o in objs):
                return None  # Already handled by object_filter
            if obj.size == min(o.size for o in objs):
                return None
            
            # Is it the most central?
            h, w = inp0.shape
            cx, cy = h // 2, w // 2
            
            def dist_to_center(o, ch=cx, cw=cy):
                r0, c0 = o.top_left
                return abs(r0 - ch) + abs(c0 - cw)
            
            if dist_to_center(obj) == min(dist_to_center(o) for o in objs):
                def keep_central(g):
                    objects = _extract_objects(g)
                    if not objects:
                        return g
                    h, w = g.shape
                    ch, cw = h // 2, w // 2
                    central = min(objects, key=lambda o: abs(o.top_left[0] - ch) + abs(o.top_left[1] - cw))
                    result = np.zeros_like(g)
                    for r, c in central.cells:
                        result[r, c] = central.color
                    return result
                if verify(keep_central, pairs):
                    return keep_central
    
    # Try: keep only objects with unique color
    color_counts = Counter(o.color for o in objs)
    unique_color_objs = [o for o in objs if color_counts[o.color] == 1]
    if unique_color_objs:
        def keep_unique_color(g):
            objects = _extract_objects(g)
            cc = Counter(o.color for o in objects)
            result = np.zeros_like(g)
            for o in objects:
                if cc[o.color] == 1:
                    for r, c in o.cells:
                        result[r, c] = o.color
            return result
        if verify(keep_unique_color, pairs):
            return keep_unique_color
    
    # Try: remove objects of the most common color
    if color_counts:
        most_common_color = color_counts.most_common(1)[0][0]
        def remove_common(g, mcc=most_common_color):
            objects = _extract_objects(g)
            cc = Counter(o.color for o in objects)
            mc = cc.most_common(1)[0][0] if cc else 0
            result = g.copy()
            for o in objects:
                if o.color == mc:
                    for r, c in o.cells:
                        result[r, c] = 0
            return result
        if verify(remove_common, pairs):
            return remove_common
    
    return None


def try_color_histogram(pairs):
    """Output encodes color frequency information."""
    inp0, out0 = pairs[0]
    oh, ow = out0.shape
    
    colors = [int(x) for x in inp0.flat if x != 0]
    if not colors:
        return None
    
    cc = Counter(colors)
    n_colors = len(cc)
    
    # N×1 output: one row per unique color, value = the color
    if oh == n_colors and ow == 1:
        sorted_c = sorted(cc.keys())
        expected = np.array([[c] for c in sorted_c], dtype=inp0.dtype)
        if np.array_equal(expected, out0):
            def unique_color_col(g):
                c = sorted(set(int(x) for x in g.flat if x != 0))
                return np.array([[x] for x in c], dtype=g.dtype)
            if verify(unique_color_col, pairs):
                return unique_color_col
    
    # 1×N output
    if oh == 1 and ow == n_colors:
        sorted_c = sorted(cc.keys())
        expected = np.array([sorted_c], dtype=inp0.dtype)
        if np.array_equal(expected, out0):
            def unique_color_row(g):
                c = sorted(set(int(x) for x in g.flat if x != 0))
                return np.array([c], dtype=g.dtype)
            if verify(unique_color_row, pairs):
                return unique_color_row
    
    return None


def try_projection(pairs):
    """Project grid onto a row or column."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    # Project to single row (OR of each column)
    if oh == 1 and ow == iw:
        def project_col(g):
            h, w = g.shape
            result = np.zeros((1, w), dtype=g.dtype)
            for c in range(w):
                for r in range(h):
                    if g[r, c] != 0:
                        result[0, c] = g[r, c]
                        break
            return result
        if verify(project_col, pairs):
            return project_col
        
        # OR projection: any non-zero in column
        def project_col_any(g):
            h, w = g.shape
            result = np.zeros((1, w), dtype=g.dtype)
            for c in range(w):
                colors = [int(g[r, c]) for r in range(h) if g[r, c] != 0]
                if colors:
                    result[0, c] = Counter(colors).most_common(1)[0][0]
            return result
        if verify(project_col_any, pairs):
            return project_col_any
    
    # Project to single column
    if oh == ih and ow == 1:
        def project_row(g):
            h, w = g.shape
            result = np.zeros((h, 1), dtype=g.dtype)
            for r in range(h):
                for c in range(w):
                    if g[r, c] != 0:
                        result[r, 0] = g[r, c]
                        break
            return result
        if verify(project_row, pairs):
            return project_row
        
        def project_row_any(g):
            h, w = g.shape
            result = np.zeros((h, 1), dtype=g.dtype)
            for r in range(h):
                colors = [int(g[r, c]) for c in range(w) if g[r, c] != 0]
                if colors:
                    result[r, 0] = Counter(colors).most_common(1)[0][0]
            return result
        if verify(project_row_any, pairs):
            return project_row_any
    
    return None


def try_subgrid_extraction(pairs):
    """Extract specific sub-grid based on markers."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    if oh >= ih or ow >= iw:
        return None
    
    # Find where output appears in input
    for r0 in range(ih - oh + 1):
        for c0 in range(iw - ow + 1):
            sub = inp0[r0:r0+oh, c0:c0+ow]
            if np.array_equal(sub, out0):
                # What distinguishes this region?
                # Check if it's the densest region
                def extract_densest(g, oh_=oh, ow_=ow):
                    h, w = g.shape
                    best = None
                    best_density = -1
                    for r in range(h - oh_ + 1):
                        for c in range(w - ow_ + 1):
                            sub = g[r:r+oh_, c:c+ow_]
                            d = np.count_nonzero(sub)
                            if d > best_density:
                                best_density = d
                                best = sub.copy()
                    return best if best is not None else g
                if verify(extract_densest, pairs):
                    return extract_densest
    
    return None


def try_repeat_pattern(pairs):
    """Repeat a detected pattern to fill the grid."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    h, w = inp0.shape
    
    # Find non-zero pattern and repeat it
    nz = np.argwhere(inp0 != 0)
    if len(nz) == 0:
        return None
    
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    ph, pw = r1 - r0 + 1, c1 - c0 + 1
    
    if ph <= 0 or pw <= 0 or ph >= h or pw >= w:
        return None
    
    pattern = inp0[r0:r0+ph, c0:c0+pw]
    
    # Check if output is this pattern tiled to fill
    if h % ph == 0 and w % pw == 0:
        tiled = np.tile(pattern, (h // ph, w // pw))
        if tiled.shape == out0.shape and np.array_equal(tiled, out0):
            def tile_pattern(g):
                nz = np.argwhere(g != 0)
                if len(nz) == 0:
                    return g
                r0, c0 = nz.min(axis=0)
                r1, c1 = nz.max(axis=0)
                pat = g[r0:r1+1, c0:c1+1]
                gh, gw = g.shape
                ph, pw = pat.shape
                if ph == 0 or pw == 0:
                    return g
                reps_h = gh // ph if gh % ph == 0 else 1
                reps_w = gw // pw if gw % pw == 0 else 1
                result = np.tile(pat, (reps_h, reps_w))
                return result[:gh, :gw]
            if verify(tile_pattern, pairs):
                return tile_pattern
    
    return None


def try_color_map_consistent(pairs):
    """Consistent pixel-level color mapping across all examples."""
    if len(pairs) < 2:
        return None
    
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    # Build color mapping from first example
    cmap = {}
    for r in range(inp0.shape[0]):
        for c in range(inp0.shape[1]):
            ic = int(inp0[r, c])
            oc = int(out0[r, c])
            if ic in cmap and cmap[ic] != oc:
                return None
            cmap[ic] = oc
    
    if all(k == v for k, v in cmap.items()):
        return None  # Identity
    
    def apply_cmap(g, cm=cmap):
        result = g.copy()
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                v = int(g[r, c])
                if v in cm:
                    result[r, c] = cm[v]
        return result
    
    if verify(apply_cmap, pairs):
        return apply_cmap
    return None


def try_border_color_fill(pairs):
    """Fill interior with border color or vice versa."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    h, w = inp0.shape
    if h < 3 or w < 3:
        return None
    
    # Get border color
    border = set()
    border.update(int(inp0[0, c]) for c in range(w))
    border.update(int(inp0[h-1, c]) for c in range(w))
    border.update(int(inp0[r, 0]) for r in range(h))
    border.update(int(inp0[r, w-1]) for r in range(h))
    
    if len(border) != 1:
        return None
    
    border_color = border.pop()
    
    # Fill interior cells that are background with border color
    def fill_interior(g, bc=border_color):
        h, w = g.shape
        result = g.copy()
        for r in range(1, h-1):
            for c in range(1, w-1):
                if result[r, c] == 0:
                    result[r, c] = bc
        return result
    if verify(fill_interior, pairs):
        return fill_interior
    
    # Remove border and keep interior
    interior = inp0[1:h-1, 1:w-1]
    if interior.shape == out0.shape and np.array_equal(interior, out0):
        def remove_border(g):
            return g[1:-1, 1:-1].copy()
        if verify(remove_border, pairs):
            return remove_border
    
    return None


def try_swap_colors(pairs):
    """Swap two specific colors."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    diff = np.where(inp0 != out0)
    if len(diff[0]) == 0:
        return None
    
    # Collect color swaps
    swaps = set()
    for r, c in zip(*diff):
        swaps.add((int(inp0[r, c]), int(out0[r, c])))
    
    if len(swaps) == 2:
        s = list(swaps)
        if s[0] == (s[1][1], s[1][0]):
            c1, c2 = s[0]
            def swap(g, a=c1, b=c2):
                result = g.copy()
                mask_a = g == a
                mask_b = g == b
                result[mask_a] = b
                result[mask_b] = a
                return result
            if verify(swap, pairs):
                return swap
    
    return None


def try_grid_concat(pairs):
    """Output is horizontal or vertical concatenation of transformed parts."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    # Output is input + flipped input (horizontal)
    if oh == ih and ow == 2 * iw:
        combos = [
            ("h_concat_fliph", lambda g: np.hstack([g, np.fliplr(g)])),
            ("h_concat_flipv", lambda g: np.hstack([g, np.flipud(g)])),
            ("h_concat_rot180", lambda g: np.hstack([g, np.rot90(g, 2)])),
            ("h_concat_self", lambda g: np.hstack([g, g])),
        ]
        for name, fn in combos:
            if verify(fn, pairs):
                return fn
    
    # Vertical
    if oh == 2 * ih and ow == iw:
        combos = [
            ("v_concat_flipv", lambda g: np.vstack([g, np.flipud(g)])),
            ("v_concat_fliph", lambda g: np.vstack([g, np.fliplr(g)])),
            ("v_concat_rot180", lambda g: np.vstack([g, np.rot90(g, 2)])),
            ("v_concat_self", lambda g: np.vstack([g, g])),
        ]
        for name, fn in combos:
            if verify(fn, pairs):
                return fn
    
    # 2×2 tiling
    if oh == 2 * ih and ow == 2 * iw:
        def tile_2x2(g):
            return np.tile(g, (2, 2))
        if verify(tile_2x2, pairs):
            return tile_2x2
        
        def tile_mirror(g):
            top = np.hstack([g, np.fliplr(g)])
            bot = np.hstack([np.flipud(g), np.rot90(g, 2)])
            return np.vstack([top, bot])
        if verify(tile_mirror, pairs):
            return tile_mirror
    
    return None


def try_fill_holes(pairs):
    """Fill holes (zeros surrounded by non-zero) with the surrounding color."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    def fill_holes(g):
        h, w = g.shape
        result = g.copy()
        # BFS from edges to find "outside" zeros
        visited = np.zeros((h, w), dtype=bool)
        queue = []
        for r in range(h):
            for c in [0, w-1]:
                if g[r, c] == 0 and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        for c in range(w):
            for r in [0, h-1]:
                if g[r, c] == 0 and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        
        # Fill interior holes with nearest non-zero color
        for r in range(h):
            for c in range(w):
                if g[r, c] == 0 and not visited[r, c]:
                    # Find surrounding color
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0:
                            result[r, c] = g[nr, nc]
                            break
        return result
    
    if verify(fill_holes, pairs):
        return fill_holes
    return None


def try_mask_by_template(pairs):
    """Use one region of the grid as a mask for another."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    # Split input horizontally, use one half as mask
    if iw % 2 == 0:
        hw = iw // 2
        left = inp0[:, :hw]
        right = inp0[:, hw:]
        
        if oh == ih and ow == hw:
            # Left where right is non-zero
            masked = np.where(right != 0, left, 0)
            if np.array_equal(masked, out0):
                def mask_lr(g):
                    hw = g.shape[1] // 2
                    return np.where(g[:, hw:] != 0, g[:, :hw], 0)
                if verify(mask_lr, pairs):
                    return mask_lr
            
            # Right where left is non-zero
            masked = np.where(left != 0, right, 0)
            if np.array_equal(masked, out0):
                def mask_rl(g):
                    hw = g.shape[1] // 2
                    return np.where(g[:, :hw] != 0, g[:, hw:], 0)
                if verify(mask_rl, pairs):
                    return mask_rl
    
    return None


ALL_EXTENDED_STRATEGIES_V3 = [
    ("cc_ops", try_connected_component_ops),
    ("color_histogram", try_color_histogram),
    ("projection", try_projection),
    ("subgrid_extract", try_subgrid_extraction),
    ("repeat_pattern", try_repeat_pattern),
    ("color_map_consistent", try_color_map_consistent),
    ("border_color_fill", try_border_color_fill),
    ("swap_colors", try_swap_colors),
    ("grid_concat", try_grid_concat),
    ("fill_holes", try_fill_holes),
    ("mask_by_template", try_mask_by_template),
]
