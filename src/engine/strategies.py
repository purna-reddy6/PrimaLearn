"""
strategies.py — Extended strategy library for ARC tasks.

Covers the major ARC task families beyond simple transforms:
- Flood fill / region filling
- Gravity / object movement  
- Grid partitioning / subgrid extraction
- Masking / template application
- Pattern denoising
- Unique/anomaly detection
- Drawing / connecting objects
- Counting-based transforms
- Pixel-level rules
- Sub-grid operations
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from src.spelke_dsl.l_objects import _extract_objects, GridObject


def verify(fn: Callable, pairs: list[tuple[np.ndarray, np.ndarray]]) -> bool:
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


# ══════════════════════════════════════════════════════════════
# FLOOD FILL / REGION FILLING
# ══════════════════════════════════════════════════════════════

def try_flood_fill(pairs):
    """Fill enclosed background regions with a color."""
    from collections import deque
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    diff = (inp0 != out0)
    if not diff.any():
        return None
    
    # Find what color fills appeared
    fill_positions = list(zip(*np.where(diff)))
    if not fill_positions:
        return None
    
    fill_color = int(out0[fill_positions[0]])
    
    # Check: are all changed cells background (0) that got filled?
    if not all(inp0[r, c] == 0 for r, c in fill_positions):
        return None
    
    def flood_fill_enclosed(g):
        h, w = g.shape
        # Find background cells reachable from border
        visited = np.zeros((h, w), dtype=bool)
        q = deque()
        for r in range(h):
            for c in [0, w-1]:
                if g[r, c] == 0 and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))
        for c in range(w):
            for r in [0, h-1]:
                if g[r, c] == 0 and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == 0:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        result = g.copy()
        for r in range(h):
            for c in range(w):
                if g[r, c] == 0 and not visited[r, c]:
                    result[r, c] = fill_color
        return result
    
    if verify(flood_fill_enclosed, pairs):
        return flood_fill_enclosed
    return None


# ══════════════════════════════════════════════════════════════
# GRAVITY / OBJECT MOVEMENT
# ══════════════════════════════════════════════════════════════

def try_gravity(pairs):
    """Move non-background pixels in a direction (gravity)."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    # Try gravity down
    def gravity_down(g):
        h, w = g.shape
        result = np.zeros_like(g)
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != 0]
            for i, v in enumerate(col):
                result[h - len(col) + i, c] = v
        return result
    
    def gravity_up(g):
        h, w = g.shape
        result = np.zeros_like(g)
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != 0]
            for i, v in enumerate(col):
                result[i, c] = v
        return result
    
    def gravity_right(g):
        h, w = g.shape
        result = np.zeros_like(g)
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != 0]
            for i, v in enumerate(row):
                result[r, w - len(row) + i] = v
        return result
    
    def gravity_left(g):
        h, w = g.shape
        result = np.zeros_like(g)
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != 0]
            for i, v in enumerate(row):
                result[r, i] = v
        return result
    
    for fn in [gravity_down, gravity_up, gravity_right, gravity_left]:
        if verify(fn, pairs):
            return fn
    return None


# ══════════════════════════════════════════════════════════════
# GRID PARTITIONING
# ══════════════════════════════════════════════════════════════

def try_grid_partition(pairs):
    """Extract a specific quadrant or subgrid."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    if oh == 0 or ow == 0:
        return None
    
    # Try quadrants
    if ih % 2 == 0 and iw % 2 == 0:
        hh, hw = ih // 2, iw // 2
        if oh == hh and ow == hw:
            quadrants = [
                ("tl", lambda g: g[:g.shape[0]//2, :g.shape[1]//2].copy()),
                ("tr", lambda g: g[:g.shape[0]//2, g.shape[1]//2:].copy()),
                ("bl", lambda g: g[g.shape[0]//2:, :g.shape[1]//2].copy()),
                ("br", lambda g: g[g.shape[0]//2:, g.shape[1]//2:].copy()),
            ]
            for name, fn in quadrants:
                if verify(fn, pairs):
                    return fn
    
    # Try thirds
    if ih % 3 == 0 and iw % 3 == 0:
        th, tw = ih // 3, iw // 3
        if oh == th and ow == tw:
            for ri in range(3):
                for ci in range(3):
                    def extract(g, r=ri, c=ci):
                        th, tw = g.shape[0]//3, g.shape[1]//3
                        return g[r*th:(r+1)*th, c*tw:(c+1)*tw].copy()
                    if verify(extract, pairs):
                        return extract
    
    # Try horizontal/vertical splits
    for div in range(2, min(ih, 6)):
        if ih % div == 0:
            sh = ih // div
            if oh == sh and ow == iw:
                for i in range(div):
                    def extract_row(g, idx=i, s=sh):
                        return g[idx*s:(idx+1)*s, :].copy()
                    if verify(extract_row, pairs):
                        return extract_row
    
    for div in range(2, min(iw, 6)):
        if iw % div == 0:
            sw = iw // div
            if oh == ih and ow == sw:
                for i in range(div):
                    def extract_col(g, idx=i, s=sw):
                        return g[:, idx*s:(idx+1)*s].copy()
                    if verify(extract_col, pairs):
                        return extract_col
    
    return None


# ══════════════════════════════════════════════════════════════
# MASKING / TEMPLATE APPLICATION
# ══════════════════════════════════════════════════════════════

def try_masking(pairs):
    """Apply one part of the grid as a mask/template to another."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    # Check if input can be split into two halves used as mask+content
    if iw % 2 == 0 and oh == ih and ow == iw // 2:
        left = inp0[:, :iw//2]
        right = inp0[:, iw//2:]
        
        # Try: output = left AND right (non-zero intersection)
        def intersect_lr(g):
            hw = g.shape[1] // 2
            l, r = g[:, :hw], g[:, hw:]
            result = np.zeros_like(l)
            mask = (l != 0) & (r != 0)
            result[mask] = l[mask]
            return result
        if verify(intersect_lr, pairs):
            return intersect_lr
        
        # Try: output = left OR right
        def union_lr(g):
            hw = g.shape[1] // 2
            l, r = g[:, :hw], g[:, hw:]
            result = l.copy()
            mask = (result == 0) & (r != 0)
            result[mask] = r[mask]
            return result
        if verify(union_lr, pairs):
            return union_lr
        
        # Try: output = left XOR right
        def xor_lr(g):
            hw = g.shape[1] // 2
            l, r = g[:, :hw], g[:, hw:]
            result = np.zeros_like(l)
            mask = (l != 0) ^ (r != 0)
            result[mask & (l != 0)] = l[mask & (l != 0)]
            result[mask & (r != 0)] = r[mask & (r != 0)]
            return result
        if verify(xor_lr, pairs):
            return xor_lr
    
    # Same for top/bottom split
    if ih % 2 == 0 and ow == iw and oh == ih // 2:
        def intersect_tb(g):
            hh = g.shape[0] // 2
            t, b = g[:hh, :], g[hh:, :]
            result = np.zeros_like(t)
            mask = (t != 0) & (b != 0)
            result[mask] = t[mask]
            return result
        if verify(intersect_tb, pairs):
            return intersect_tb
        
        def union_tb(g):
            hh = g.shape[0] // 2
            t, b = g[:hh, :], g[hh:, :]
            result = t.copy()
            mask = (result == 0) & (b != 0)
            result[mask] = b[mask]
            return result
        if verify(union_tb, pairs):
            return union_tb
    
    return None


# ══════════════════════════════════════════════════════════════
# DENOISING / MOST COMMON PATTERN
# ══════════════════════════════════════════════════════════════

def try_denoise(pairs):
    """Remove noise pixels — keep only majority color per object or region."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    # Try: remove color that appears least (replace with 0)
    from collections import Counter
    colors_in = Counter(int(x) for x in inp0.flat if x != 0)
    colors_out = Counter(int(x) for x in out0.flat if x != 0)
    
    removed = set(colors_in.keys()) - set(colors_out.keys())
    if len(removed) == 1:
        noise_color = removed.pop()
        def remove_color(g, nc=noise_color):
            result = g.copy()
            result[result == nc] = 0
            return result
        if verify(remove_color, pairs):
            return remove_color
    
    # Try: keep only most common non-bg color
    if colors_out and len(colors_out) < len(colors_in):
        keep = set(colors_out.keys())
        def keep_colors(g, k=keep):
            result = g.copy()
            for r in range(g.shape[0]):
                for c in range(g.shape[1]):
                    if int(g[r, c]) not in k and g[r, c] != 0:
                        result[r, c] = 0
            return result
        if verify(keep_colors, pairs):
            return keep_colors
    
    return None


# ══════════════════════════════════════════════════════════════
# UNIQUE / ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════

def try_find_unique(pairs):
    """Find the unique/different object among similar ones."""
    inp0, out0 = pairs[0]
    objs = _extract_objects(inp0)
    if len(objs) < 3:
        return None
    
    # Group objects by shape
    from collections import defaultdict
    shape_groups = defaultdict(list)
    for o in objs:
        key = (o.mask.tobytes(), o.mask.shape)
        shape_groups[key].append(o)
    
    # Find the unique one (appears once while others appear multiple times)
    unique_objs = [objs_list[0] for objs_list in shape_groups.values() if len(objs_list) == 1]
    if len(unique_objs) == 1:
        uo = unique_objs[0]
        ug = uo.to_grid()
        if ug.shape == out0.shape and np.array_equal(ug, out0):
            def extract_unique(g):
                objects = _extract_objects(g)
                if len(objects) < 3:
                    return g
                groups = defaultdict(list)
                for o in objects:
                    key = (o.mask.tobytes(), o.mask.shape)
                    groups[key].append(o)
                uniques = [ol[0] for ol in groups.values() if len(ol) == 1]
                if uniques:
                    return uniques[0].to_grid()
                return g
            if verify(extract_unique, pairs):
                return extract_unique
    return None


# ══════════════════════════════════════════════════════════════
# COUNTING-BASED TRANSFORMS
# ══════════════════════════════════════════════════════════════

def try_count_based(pairs):
    """Output depends on counting objects/colors."""
    inp0, out0 = pairs[0]
    objs = _extract_objects(inp0)
    n_colors = len(set(int(x) for x in inp0.flat if x != 0))
    n_objs = len(objs)
    
    # Output is n×n grid of a single color
    if out0.shape[0] == out0.shape[1] and len(set(out0.flat)) <= 2:
        size = out0.shape[0]
        color = int(out0[0, 0]) if out0[0, 0] != 0 else int(out0.max())
        
        if size == n_objs:
            def count_to_grid(g, c=color):
                objects = _extract_objects(g)
                n = len(objects)
                return np.full((n, n), c, dtype=g.dtype)
            if verify(count_to_grid, pairs):
                return count_to_grid
        
        if size == n_colors:
            def colors_to_grid(g, c=color):
                nc = len(set(int(x) for x in g.flat if x != 0))
                return np.full((nc, nc), c, dtype=g.dtype)
            if verify(colors_to_grid, pairs):
                return colors_to_grid
    
    # Output is 1×n or n×1 encoding counts
    if out0.shape[0] == 1 or out0.shape[1] == 1:
        vals = out0.flatten()
        if len(vals) == n_colors and all(v > 0 for v in vals):
            pass  # Complex counting — skip for now
    
    return None


# ══════════════════════════════════════════════════════════════
# PIXEL-LEVEL CELLULAR RULES
# ══════════════════════════════════════════════════════════════

def try_pixel_rules(pairs):
    """Detect simple pixel-level rules (neighbor-based)."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    h, w = inp0.shape
    
    # Try: each cell becomes the most common neighbor color
    def majority_neighbors(g):
        from collections import Counter
        h, w = g.shape
        result = g.copy()
        for r in range(h):
            for c in range(w):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(int(g[nr, nc]))
                if neighbors:
                    cnt = Counter(neighbors)
                    most = cnt.most_common(1)[0][0]
                    if g[r, c] == 0 and most != 0:
                        result[r, c] = most
        return result
    
    if verify(majority_neighbors, pairs):
        return majority_neighbors
    
    return None


# ══════════════════════════════════════════════════════════════
# OBJECT DUPLICATION / STAMPING
# ══════════════════════════════════════════════════════════════

def try_stamp_pattern(pairs):
    """Stamp/duplicate a pattern at marked locations."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    objs = _extract_objects(inp0)
    if len(objs) < 2:
        return None
    
    # Find the "template" (smallest object) and "markers" (single cells)
    by_size = sorted(objs, key=lambda o: o.size)
    markers = [o for o in by_size if o.size == 1]
    templates = [o for o in by_size if o.size > 1]
    
    if not markers or not templates:
        return None
    
    template = templates[0]
    tgrid = template.to_grid()
    th, tw = tgrid.shape
    
    def stamp(g):
        objects = _extract_objects(g)
        mk = [o for o in objects if o.size == 1]
        tpl = [o for o in objects if o.size > 1]
        if not mk or not tpl:
            return g
        t = sorted(tpl, key=lambda o: o.size)[0]
        tg = t.to_grid()
        result = g.copy()
        for m in mk:
            mr, mc = list(m.cells)[0]
            for r in range(tg.shape[0]):
                for c in range(tg.shape[1]):
                    if tg[r, c] != 0:
                        pr, pc = mr + r - tg.shape[0]//2, mc + c - tg.shape[1]//2
                        if 0 <= pr < result.shape[0] and 0 <= pc < result.shape[1]:
                            result[pr, pc] = tg[r, c]
        return result
    
    if verify(stamp, pairs):
        return stamp
    return None


# ══════════════════════════════════════════════════════════════
# CROP TO NON-ZERO BOUNDING BOX
# ══════════════════════════════════════════════════════════════

def try_crop_nonzero(pairs):
    """Crop to the bounding box of all non-zero cells."""
    inp0, out0 = pairs[0]
    
    nonzero = np.argwhere(inp0 != 0)
    if len(nonzero) == 0:
        return None
    
    r0, c0 = nonzero.min(axis=0)
    r1, c1 = nonzero.max(axis=0)
    cropped = inp0[r0:r1+1, c0:c1+1]
    
    if cropped.shape == out0.shape and np.array_equal(cropped, out0):
        def crop_nz(g):
            nz = np.argwhere(g != 0)
            if len(nz) == 0:
                return g
            r0, c0 = nz.min(axis=0)
            r1, c1 = nz.max(axis=0)
            return g[r0:r1+1, c0:c1+1].copy()
        if verify(crop_nz, pairs):
            return crop_nz
    return None


# ══════════════════════════════════════════════════════════════
# MIRROR / REFLECT WITHIN GRID
# ══════════════════════════════════════════════════════════════

def try_mirror_extend(pairs):
    """Extend grid by mirroring (e.g., double width with mirror)."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    # Double width with horizontal mirror
    if oh == ih and ow == 2 * iw:
        def mirror_h(g):
            return np.hstack([g, np.fliplr(g)])
        if verify(mirror_h, pairs):
            return mirror_h
        def mirror_h2(g):
            return np.hstack([np.fliplr(g), g])
        if verify(mirror_h2, pairs):
            return mirror_h2
    
    # Double height with vertical mirror
    if oh == 2 * ih and ow == iw:
        def mirror_v(g):
            return np.vstack([g, np.flipud(g)])
        if verify(mirror_v, pairs):
            return mirror_v
        def mirror_v2(g):
            return np.vstack([np.flipud(g), g])
        if verify(mirror_v2, pairs):
            return mirror_v2
    
    # 2x2 tiling with mirrors
    if oh == 2 * ih and ow == 2 * iw:
        combos = [
            lambda g: np.vstack([np.hstack([g, np.fliplr(g)]), np.hstack([np.flipud(g), np.flipud(np.fliplr(g))])]),
            lambda g: np.vstack([np.hstack([g, np.fliplr(g)]), np.hstack([np.flipud(g), np.rot90(g, 2)])]),
            lambda g: np.tile(g, (2, 2)),
        ]
        for fn in combos:
            if verify(fn, pairs):
                return fn
    
    return None


# ══════════════════════════════════════════════════════════════
# COLOR MAPPING BY POSITION / MAJORITY
# ══════════════════════════════════════════════════════════════

def try_majority_color(pairs):
    """Replace each object's color with the majority color in the grid."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    from collections import Counter
    
    # Most common non-bg color
    colors = Counter(int(x) for x in inp0.flat if x != 0)
    if not colors:
        return None
    majority = colors.most_common(1)[0][0]
    
    # Check if output replaces all non-bg with majority
    def to_majority(g):
        result = g.copy()
        result[result != 0] = majority
        return result
    if verify(to_majority, pairs):
        return to_majority
    
    # Check if output replaces bg with majority where neighbors exist
    return None


# ══════════════════════════════════════════════════════════════
# SORT ROWS/COLUMNS
# ══════════════════════════════════════════════════════════════

def try_sort_grid(pairs):
    """Sort rows or columns by some criterion."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    h, w = inp0.shape
    
    # Try: sort rows by number of non-zero cells (ascending)
    def sort_rows_asc(g):
        rows = [g[r, :].copy() for r in range(g.shape[0])]
        rows.sort(key=lambda r: np.count_nonzero(r))
        return np.array(rows)
    if verify(sort_rows_asc, pairs):
        return sort_rows_asc
    
    def sort_rows_desc(g):
        rows = [g[r, :].copy() for r in range(g.shape[0])]
        rows.sort(key=lambda r: -np.count_nonzero(r))
        return np.array(rows)
    if verify(sort_rows_desc, pairs):
        return sort_rows_desc
    
    # Sort columns
    def sort_cols_asc(g):
        cols = [g[:, c].copy() for c in range(g.shape[1])]
        cols.sort(key=lambda c: np.count_nonzero(c))
        return np.column_stack(cols)
    if verify(sort_cols_asc, pairs):
        return sort_cols_asc
    
    return None


# ══════════════════════════════════════════════════════════════
# BOOLEAN OPERATIONS ON GRID HALVES
# ══════════════════════════════════════════════════════════════

def try_boolean_grid_ops(pairs):
    """Combine grid halves with boolean operations."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    # Same size output — might be overlay of objects
    if inp0.shape == out0.shape:
        objs = _extract_objects(inp0)
        if len(objs) == 2:
            o1, o2 = objs[0], objs[1]
            # Try: keep only overlapping region
            overlap = o1.cells & o2.cells
            if overlap:
                def keep_overlap(g):
                    objects = _extract_objects(g)
                    if len(objects) != 2:
                        return g
                    ol = objects[0].cells & objects[1].cells
                    result = np.zeros_like(g)
                    for r, c in ol:
                        result[r, c] = g[r, c]
                    return result
                if verify(keep_overlap, pairs):
                    return keep_overlap
    return None


# ══════════════════════════════════════════════════════════════  
# COLLECT ALL STRATEGIES
# ══════════════════════════════════════════════════════════════

ALL_EXTENDED_STRATEGIES = [
    ("flood_fill", try_flood_fill),
    ("gravity", try_gravity),
    ("grid_partition", try_grid_partition),
    ("masking", try_masking),
    ("denoise", try_denoise),
    ("find_unique", try_find_unique),
    ("count_based", try_count_based),
    ("pixel_rules", try_pixel_rules),
    ("stamp_pattern", try_stamp_pattern),
    ("crop_nonzero", try_crop_nonzero),
    ("mirror_extend", try_mirror_extend),
    ("majority_color", try_majority_color),
    ("sort_grid", try_sort_grid),
    ("boolean_grid", try_boolean_grid_ops),
]
