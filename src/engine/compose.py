"""
compose.py — Compositional search: 3-7 step program synthesis.

The key missing piece: instead of single-strategy matching,
systematically explore CHAINS of primitives:
  extract_objects → filter → transform → recolor → place_back

Uses the recognition model to prioritize which chains to try.
"""

from __future__ import annotations
import signal
import numpy as np
from typing import Callable, Optional
from src.spelke_dsl.l_objects import _extract_objects, GridObject


class _Timeout(Exception):
    pass

def _alarm(sig, frame):
    raise _Timeout()


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


# ═══════════════════════════════════════════════════════════
# ATOMIC OPERATIONS for composition
# ═══════════════════════════════════════════════════════════

def _op_extract_objects(g):
    return _extract_objects(g)

def _op_extract_objects_all(g):
    """Extract including background-colored objects."""
    return _extract_objects(g)

def _op_largest(objs):
    return max(objs, key=lambda o: o.size) if objs else None

def _op_smallest(objs):
    return min(objs, key=lambda o: o.size) if objs else None

def _op_to_grid(obj):
    return obj.to_grid() if obj else None

def _op_rotate90(g):
    return np.rot90(g, k=-1).copy()

def _op_rotate180(g):
    return np.rot90(g, k=2).copy()

def _op_flip_h(g):
    return np.fliplr(g).copy()

def _op_flip_v(g):
    return np.flipud(g).copy()

def _op_transpose(g):
    return g.T.copy()

def _op_crop_nonzero(g):
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return g
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return g[r0:r1+1, c0:c1+1].copy()

def _op_scale_up(g, factor):
    return np.repeat(np.repeat(g, factor, axis=0), factor, axis=1)

def _op_scale_down(g, factor):
    if factor <= 0:
        return g
    return g[::factor, ::factor].copy()


# ═══════════════════════════════════════════════════════════
# 2-STEP COMPOSITIONS: Object → Transform → Output
# ═══════════════════════════════════════════════════════════

def try_obj_transform_chains(pairs):
    """Extract object, apply transform, output result."""
    inp0, out0 = pairs[0]
    
    try:
        in_objs = _extract_objects(inp0)
    except Exception:
        return None
    
    if not in_objs:
        return None
    
    # Object selectors
    selectors = [
        ("largest", lambda objs: max(objs, key=lambda o: o.size) if objs else None),
        ("smallest", lambda objs: min(objs, key=lambda o: o.size) if objs else None),
    ]
    
    # Add color-specific selectors
    for obj in in_objs:
        c = obj.color
        selectors.append((f"color_{c}", lambda objs, col=c: next((o for o in objs if o.color == col), None)))
    
    # Transforms on grid
    transforms = [
        ("identity", lambda g: g),
        ("rot90", lambda g: np.rot90(g, k=-1).copy()),
        ("rot180", lambda g: np.rot90(g, k=2).copy()),
        ("flip_h", lambda g: np.fliplr(g).copy()),
        ("flip_v", lambda g: np.flipud(g).copy()),
        ("transpose", lambda g: g.T.copy()),
    ]
    
    for sel_name, selector in selectors:
        for tf_name, transform in transforms:
            def chain(g, sel=selector, tf=transform):
                objs = _extract_objects(g)
                obj = sel(objs)
                if obj is None:
                    return g
                og = obj.to_grid()
                return tf(og)
            
            if verify(chain, pairs):
                return chain
    
    return None


# ═══════════════════════════════════════════════════════════
# 3-STEP: Extract → Filter → Combine
# ═══════════════════════════════════════════════════════════

def try_filter_combine(pairs):
    """Extract objects, filter, combine back."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    try:
        in_objs = _extract_objects(inp0)
    except Exception:
        return None
    
    if len(in_objs) < 2:
        return None
    
    # Try: keep only objects of specific sizes
    out_objs = _extract_objects(out0)
    if not out_objs:
        return None
    
    kept_sizes = set(o.size for o in out_objs)
    removed_sizes = set(o.size for o in in_objs) - kept_sizes
    
    if removed_sizes:
        def filter_by_size(g, ks=kept_sizes):
            objs = _extract_objects(g)
            result = np.zeros_like(g)
            for o in objs:
                if o.size in ks:
                    for r, c in o.cells:
                        result[r, c] = o.color
            return result
        if verify(filter_by_size, pairs):
            return filter_by_size
    
    # Try: keep objects above/below median size
    if len(in_objs) >= 3:
        sizes = sorted(set(o.size for o in in_objs))
        median = sizes[len(sizes)//2]
        
        def keep_above_median(g, m=median):
            objs = _extract_objects(g)
            result = np.zeros_like(g)
            for o in objs:
                if o.size >= m:
                    for r, c in o.cells:
                        result[r, c] = o.color
            return result
        if verify(keep_above_median, pairs):
            return keep_above_median
        
        def keep_below_median(g, m=median):
            objs = _extract_objects(g)
            result = np.zeros_like(g)
            for o in objs:
                if o.size <= m:
                    for r, c in o.cells:
                        result[r, c] = o.color
            return result
        if verify(keep_below_median, pairs):
            return keep_below_median
    
    return None


# ═══════════════════════════════════════════════════════════
# 3-STEP: Extract → Transform Each → Reassemble
# ═══════════════════════════════════════════════════════════

def try_transform_each_object(pairs):
    """Apply a transform to each object individually."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    try:
        in_objs = _extract_objects(inp0)
    except Exception:
        return None
    
    if len(in_objs) < 2:
        return None
    
    transforms = [
        ("rot90", lambda mask: np.rot90(mask, k=-1)),
        ("rot180", lambda mask: np.rot90(mask, k=2)),
        ("flip_h", lambda mask: np.fliplr(mask)),
        ("flip_v", lambda mask: np.flipud(mask)),
    ]
    
    for tf_name, tf in transforms:
        def apply_each(g, transform=tf):
            h, w = g.shape
            result = np.zeros_like(g)
            objs = _extract_objects(g)
            for o in objs:
                r0, c0, r1, c1 = o.bbox
                mask = o.mask
                try:
                    new_mask = transform(mask)
                except Exception:
                    continue
                nh, nw = new_mask.shape
                # Center on same position
                cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
                nr0 = cr - nh // 2
                nc0 = cc - nw // 2
                for r in range(nh):
                    for c in range(nw):
                        if new_mask[r, c]:
                            pr, pc = nr0 + r, nc0 + c
                            if 0 <= pr < h and 0 <= pc < w:
                                result[pr, pc] = o.color
            return result
        
        if verify(apply_each, pairs):
            return apply_each
    
    return None


# ═══════════════════════════════════════════════════════════
# Object → Crop → Scale → Output
# ═══════════════════════════════════════════════════════════

def try_crop_and_scale(pairs):
    """Crop to object then scale to output size."""
    inp0, out0 = pairs[0]
    oh, ow = out0.shape
    
    try:
        in_objs = _extract_objects(inp0)
    except Exception:
        return None
    
    for obj in in_objs:
        og = obj.to_grid()
        gh, gw = og.shape
        if gh == 0 or gw == 0:
            continue
        
        # Check if output is scaled version of this object's grid
        if oh % gh == 0 and ow % gw == 0:
            sr, sc = oh // gh, ow // gw
            if sr == sc and sr > 1:
                scaled = np.repeat(np.repeat(og, sr, axis=0), sr, axis=1)
                if scaled.shape == out0.shape and np.array_equal(scaled, out0):
                    obj_color = obj.color
                    factor = sr
                    
                    def crop_scale(g, oc=obj_color, f=factor):
                        objs = _extract_objects(g)
                        target = None
                        for o in objs:
                            if o.color == oc:
                                target = o
                                break
                        if target is None and objs:
                            target = max(objs, key=lambda o: o.size)
                        if target is None:
                            return g
                        tg = target.to_grid()
                        return np.repeat(np.repeat(tg, f, axis=0), f, axis=1)
                    
                    if verify(crop_scale, pairs):
                        return crop_scale
    
    return None


# ═══════════════════════════════════════════════════════════
# Overlay two transforms
# ═══════════════════════════════════════════════════════════

def try_overlay_transforms(pairs):
    """Output = overlay of input with transformed version."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    transforms = [
        ("rot90", lambda g: np.rot90(g, k=-1).copy()),
        ("rot180", lambda g: np.rot90(g, k=2).copy()),
        ("flip_h", lambda g: np.fliplr(g).copy()),
        ("flip_v", lambda g: np.flipud(g).copy()),
    ]
    
    for t1_name, t1 in transforms:
        for t2_name, t2 in transforms:
            if t1_name == t2_name:
                continue
            
            def overlay2(g, a=t1, b=t2):
                r1 = a(g)
                r2 = b(g)
                result = g.copy()
                mask1 = r1 != 0
                mask2 = r2 != 0
                result[mask1] = r1[mask1]
                result[mask2] = r2[mask2]
                return result
            
            if verify(overlay2, pairs):
                return overlay2
    
    # Input + one transform overlay
    for t_name, t in transforms:
        def overlay1(g, tf=t):
            result = g.copy()
            t_result = tf(g)
            mask = (t_result != 0) & (result == 0)
            result[mask] = t_result[mask]
            return result
        
        if verify(overlay1, pairs):
            return overlay1
    
    return None


# ═══════════════════════════════════════════════════════════
# Color swap based on object relationships
# ═══════════════════════════════════════════════════════════

def try_relational_recolor(pairs):
    """Recolor based on relative properties (largest gets color X, etc.)."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    
    try:
        in_objs = _extract_objects(inp0)
        out_objs = _extract_objects(out0)
    except Exception:
        return None
    
    if len(in_objs) != len(out_objs) or len(in_objs) < 2:
        return None
    
    # Match objects by cells
    matched = []
    for io in in_objs:
        for oo in out_objs:
            if io.cells == oo.cells:
                matched.append((io, oo))
                break
    
    if len(matched) != len(in_objs):
        return None
    
    # Check: sorted by size, colors follow a specific pattern
    sorted_by_size = sorted(matched, key=lambda x: x[0].size)
    rank_to_color = {i: oo.color for i, (io, oo) in enumerate(sorted_by_size)}
    
    def recolor_by_rank(g, rtc=rank_to_color):
        objs = _extract_objects(g)
        sorted_objs = sorted(objs, key=lambda o: o.size)
        result = g.copy()
        for i, o in enumerate(sorted_objs):
            if i in rtc:
                for r, c in o.cells:
                    result[r, c] = rtc[i]
        return result
    
    if verify(recolor_by_rank, pairs):
        return recolor_by_rank
    return None


# ═══════════════════════════════════════════════════════════
# Grid arithmetic: XOR, AND, DIFF of input regions
# ═══════════════════════════════════════════════════════════

def try_grid_arithmetic(pairs):
    """Combine grid regions with arithmetic operations."""
    inp0, out0 = pairs[0]
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    # Horizontal split → operation
    if iw % 2 == 0:
        hw = iw // 2
        if oh == ih and ow == hw:
            left = inp0[:, :hw]
            right = inp0[:, hw:]
            
            ops = [
                # Diff: where they differ
                lambda l, r: np.where(l != r, np.maximum(l, r), 0),
                # AND: where both non-zero
                lambda l, r: np.where((l != 0) & (r != 0), l, 0),
                # OR
                lambda l, r: np.where(l != 0, l, r),
                # XOR: one but not both
                lambda l, r: np.where((l != 0) ^ (r != 0), np.maximum(l, r), 0),
                # Right where left is zero
                lambda l, r: np.where((l == 0) & (r != 0), r, 0),
                # Left where right is zero
                lambda l, r: np.where((l != 0) & (r == 0), l, 0),
            ]
            
            for op in ops:
                def apply_op(g, operation=op):
                    hw = g.shape[1] // 2
                    return operation(g[:, :hw], g[:, hw:])
                if verify(apply_op, pairs):
                    return apply_op
    
    # Vertical split
    if ih % 2 == 0:
        hh = ih // 2
        if oh == hh and ow == iw:
            top = inp0[:hh, :]
            bot = inp0[hh:, :]
            
            ops = [
                lambda t, b: np.where(t != b, np.maximum(t, b), 0),
                lambda t, b: np.where((t != 0) & (b != 0), t, 0),
                lambda t, b: np.where(t != 0, t, b),
                lambda t, b: np.where((t != 0) ^ (b != 0), np.maximum(t, b), 0),
                lambda t, b: np.where((t == 0) & (b != 0), b, 0),
                lambda t, b: np.where((t != 0) & (b == 0), t, 0),
            ]
            
            for op in ops:
                def apply_op(g, operation=op):
                    hh = g.shape[0] // 2
                    return operation(g[:hh, :], g[hh:, :])
                if verify(apply_op, pairs):
                    return apply_op
    
    return None


# ═══════════════════════════════════════════════════════════
# Fill pattern from seed
# ═══════════════════════════════════════════════════════════

def try_line_extension(pairs):
    """Extend colored cells along lines (h/v)."""
    inp0, out0 = pairs[0]
    if inp0.shape != out0.shape:
        return None
    h, w = inp0.shape
    
    # Try: extend each non-zero cell horizontally to fill row
    def extend_h(g):
        result = g.copy()
        for r in range(g.shape[0]):
            colors = [int(g[r, c]) for c in range(g.shape[1]) if g[r, c] != 0]
            if len(colors) == 1:
                for c in range(g.shape[1]):
                    result[r, c] = colors[0]
        return result
    if verify(extend_h, pairs):
        return extend_h
    
    # Try: extend vertically
    def extend_v(g):
        result = g.copy()
        for c in range(g.shape[1]):
            colors = [int(g[r, c]) for r in range(g.shape[0]) if g[r, c] != 0]
            if len(colors) == 1:
                for r in range(g.shape[0]):
                    result[r, c] = colors[0]
        return result
    if verify(extend_v, pairs):
        return extend_v
    
    # Try: fill row with its single non-zero color
    def fill_row_color(g):
        result = g.copy()
        for r in range(g.shape[0]):
            unique = set(int(g[r, c]) for c in range(g.shape[1])) - {0}
            if len(unique) == 1:
                color = unique.pop()
                for c in range(g.shape[1]):
                    if result[r, c] == 0:
                        result[r, c] = color
        return result
    if verify(fill_row_color, pairs):
        return fill_row_color
    
    return None


# ═══════════════════════════════════════════════════════════
# Count-based output
# ═══════════════════════════════════════════════════════════

def try_count_output(pairs):
    """Output encodes counts of objects/colors."""
    inp0, out0 = pairs[0]
    
    # Try: output is 1×N where each cell = count of that color
    try:
        in_objs = _extract_objects(inp0)
    except Exception:
        return None
    
    from collections import Counter
    color_counts = Counter(int(x) for x in inp0.flat if x != 0)
    
    # Try: output row = colors sorted by frequency
    if out0.shape[0] == 1:
        vals = [int(x) for x in out0[0]]
        sorted_colors = [c for c, _ in color_counts.most_common()]
        if vals == sorted_colors:
            def sorted_color_row(g):
                cc = Counter(int(x) for x in g.flat if x != 0)
                sc = [c for c, _ in cc.most_common()]
                return np.array([sc], dtype=g.dtype)
            if verify(sorted_color_row, pairs):
                return sorted_color_row
    
    # Output is colored bars representing counts
    if out0.shape[1] == len(color_counts) and out0.shape[0] > 0:
        max_count = max(color_counts.values()) if color_counts else 0
        if out0.shape[0] == max_count:
            def count_bars(g):
                cc = Counter(int(x) for x in g.flat if x != 0)
                if not cc:
                    return g
                mc = max(cc.values())
                cols = sorted(cc.keys())
                result = np.zeros((mc, len(cols)), dtype=g.dtype)
                for ci, color in enumerate(cols):
                    for r in range(cc[color]):
                        result[mc - 1 - r, ci] = color
                return result
            if verify(count_bars, pairs):
                return count_bars
    
    return None


ALL_COMPOSITIONAL_STRATEGIES = [
    ("obj_transform_chain", try_obj_transform_chains),
    ("filter_combine", try_filter_combine),
    ("transform_each", try_transform_each_object),
    ("crop_and_scale", try_crop_and_scale),
    ("overlay_transforms", try_overlay_transforms),
    ("relational_recolor", try_relational_recolor),
    ("grid_arithmetic", try_grid_arithmetic),
    ("line_extension", try_line_extension),
    ("count_output", try_count_output),
]
