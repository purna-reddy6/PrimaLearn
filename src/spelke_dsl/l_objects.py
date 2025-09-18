"""
l_objects.py — Core System 1: Objects

Implements Spelke's object principles: cohesion, continuity, contact, persistence.
Objects are bounded, solid, persist when occluded, move on continuous paths.

Evidence base: Baillargeon drawbridge studies (3-4 months); Spelke habituation paradigm;
Feigenson & Carey 2003 parallel-individuation (cap ~3 objects).

For ARC-AGI: objects are connected components of same-color cells in 2D grids.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Any
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tpoint, tdir, tlist, tpair, trect,
)


# ──────────────────────────────────────────────────────────────────────
# Object representation
# ──────────────────────────────────────────────────────────────────────

class GridObject:
    """An extracted object from a grid — a connected component."""
    __slots__ = ("cells", "color", "bbox", "grid_shape", "_mask")

    def __init__(self, cells: set[tuple[int, int]], color: int, grid_shape: tuple[int, int]):
        self.cells = frozenset(cells)
        self.color = color
        self.grid_shape = grid_shape
        r = [c[0] for c in cells]
        c = [c[1] for c in cells]
        self.bbox = (min(r), min(c), max(r), max(c))
        self._mask = None

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            r0, c0, r1, c1 = self.bbox
            m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in self.cells:
                m[r - r0, c - c0] = True
            self._mask = m
        return self._mask

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def center(self) -> tuple[float, float]:
        rs = [c[0] for c in self.cells]
        cs = [c[1] for c in self.cells]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    @property
    def top_left(self) -> tuple[int, int]:
        return self.bbox[0], self.bbox[1]

    def to_grid(self, background: int = 0) -> np.ndarray:
        """Render this object as a minimal grid."""
        r0, c0, r1, c1 = self.bbox
        g = np.full((r1 - r0 + 1, c1 - c0 + 1), background, dtype=np.int8)
        for r, c in self.cells:
            g[r - r0, c - c0] = self.color
        return g

    def translate(self, dr: int, dc: int) -> GridObject:
        new_cells = {(r + dr, c + dc) for r, c in self.cells}
        return GridObject(new_cells, self.color, self.grid_shape)

    def __eq__(self, other):
        if not isinstance(other, GridObject):
            return False
        return self.cells == other.cells and self.color == other.color

    def __hash__(self):
        return hash((self.cells, self.color))

    def __repr__(self):
        return f"Obj(color={self.color}, size={self.size}, bbox={self.bbox})"


# ──────────────────────────────────────────────────────────────────────
# Primitive implementations
# ──────────────────────────────────────────────────────────────────────

def _extract_objects(grid: np.ndarray, background: int = 0,
                     connectivity: int = 4) -> list[GridObject]:
    """
    COHESION primitive: extract connected components from a grid.
    4-connectivity by default (Spelke's cohesion = bounded, connected).
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(h):
        for j in range(w):
            if visited[i, j] or grid[i, j] == background:
                continue
            color = int(grid[i, j])
            cells = set()
            queue = deque([(i, j)])
            visited[i, j] = True
            while queue:
                r, c = queue.popleft()
                cells.add((r, c))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if grid[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            objects.append(GridObject(cells, color, (h, w)))
    return objects


def _extract_objects_multicolor(grid: np.ndarray, background: int = 0) -> list[GridObject]:
    """Extract objects ignoring color — groups any non-background connected cells."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(h):
        for j in range(w):
            if visited[i, j] or grid[i, j] == background:
                continue
            cells = set()
            queue = deque([(i, j)])
            visited[i, j] = True
            while queue:
                r, c = queue.popleft()
                cells.add((r, c))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if grid[nr, nc] != background:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            # Use most common color
            colors = [grid[r, c] for r, c in cells]
            from collections import Counter
            color = Counter(colors).most_common(1)[0][0]
            objects.append(GridObject(cells, int(color), (h, w)))
    return objects


def _object_persistence(obj: GridObject, grid1: np.ndarray,
                        grid2: np.ndarray) -> GridObject | None:
    """
    PERSISTENCE: find the same object across two grids.
    Identity by shape match (translation-invariant).
    """
    objs2 = _extract_objects(grid2, background=0)
    normalized = obj.mask
    for o2 in objs2:
        if o2.color == obj.color and o2.mask.shape == normalized.shape:
            if np.array_equal(o2.mask, normalized):
                return o2
    return None


def _touching(obj1: GridObject, obj2: GridObject) -> bool:
    """CONTACT: are two objects adjacent (sharing a 4-connected boundary)?"""
    for r, c in obj1.cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in obj2.cells:
                return True
    return False


def _overlapping(obj1: GridObject, obj2: GridObject) -> bool:
    """Do two objects share any cells?"""
    return bool(obj1.cells & obj2.cells)


def _z_order(objects: list[GridObject]) -> list[GridObject]:
    """
    OCCLUSION: order objects by implied z-depth.
    Larger objects are assumed behind smaller overlapping ones.
    """
    return sorted(objects, key=lambda o: o.size, reverse=True)


def _copy_object(obj: GridObject) -> GridObject:
    """Create a copy of an object."""
    return GridObject(set(obj.cells), obj.color, obj.grid_shape)


def _translate_object(obj: GridObject, dr: int, dc: int) -> GridObject:
    """CONTINUITY: move an object by (dr, dc)."""
    return obj.translate(dr, dc)


def _rotate_object_90(obj: GridObject) -> GridObject:
    """Rotate an object 90 degrees clockwise around its center."""
    cr, cc = obj.center
    new_cells = set()
    for r, c in obj.cells:
        nr = int(round(cc - c + cr))
        nc = int(round(r - cr + cc))
        new_cells.add((nr, nc))
    # Re-normalize to positive coordinates
    min_r = min(r for r, c in new_cells)
    min_c = min(c for r, c in new_cells)
    if min_r < 0 or min_c < 0:
        new_cells = {(r - min_r, c - min_c) for r, c in new_cells}
    return GridObject(new_cells, obj.color, obj.grid_shape)


def _reflect_object_h(obj: GridObject) -> GridObject:
    """Reflect object horizontally (flip left-right)."""
    r0, c0, r1, c1 = obj.bbox
    new_cells = {(r, c1 - (c - c0)) for r, c in obj.cells}
    return GridObject(new_cells, obj.color, obj.grid_shape)


def _reflect_object_v(obj: GridObject) -> GridObject:
    """Reflect object vertically (flip top-bottom)."""
    r0, c0, r1, c1 = obj.bbox
    new_cells = {(r1 - (r - r0), c) for r, c in obj.cells}
    return GridObject(new_cells, obj.color, obj.grid_shape)


def _bbox(obj: GridObject) -> tuple[int, int, int, int]:
    """Get bounding box (r_min, c_min, r_max, c_max)."""
    return obj.bbox


def _crop_grid(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """Crop a grid region."""
    return grid[r0:r1 + 1, c0:c1 + 1].copy()


def _place_object(grid: np.ndarray, obj: GridObject) -> np.ndarray:
    """Place an object onto a grid, returning a new grid."""
    g = grid.copy()
    for r, c in obj.cells:
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            g[r, c] = obj.color
    return g


def _recolor_object(obj: GridObject, new_color: int) -> GridObject:
    """Change the color of an object."""
    return GridObject(set(obj.cells), new_color, obj.grid_shape)


def _object_color(obj: GridObject) -> int:
    return obj.color


def _object_size(obj: GridObject) -> int:
    return obj.size


def _object_position(obj: GridObject) -> tuple[int, int]:
    return obj.top_left


def _objects_same_shape(o1: GridObject, o2: GridObject) -> bool:
    """Shape comparison (translation-invariant)."""
    if o1.mask.shape != o2.mask.shape:
        return False
    return np.array_equal(o1.mask, o2.mask)


def _filter_by_color(objects: list[GridObject], color: int) -> list[GridObject]:
    return [o for o in objects if o.color == color]


def _filter_by_size(objects: list[GridObject], size: int) -> list[GridObject]:
    return [o for o in objects if o.size == size]


def _largest_object(objects: list[GridObject]) -> GridObject | None:
    if not objects:
        return None
    return max(objects, key=lambda o: o.size)


def _smallest_object(objects: list[GridObject]) -> GridObject | None:
    if not objects:
        return None
    return min(objects, key=lambda o: o.size)


def _sort_objects_by_size(objects: list[GridObject]) -> list[GridObject]:
    return sorted(objects, key=lambda o: o.size)


def _sort_objects_by_position(objects: list[GridObject]) -> list[GridObject]:
    return sorted(objects, key=lambda o: (o.top_left[0], o.top_left[1]))


def _grid_colors(grid: np.ndarray) -> list[int]:
    """Get unique non-background colors in the grid."""
    return sorted(set(int(x) for x in grid.flat if x != 0))


def _fill_rect(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int,
               color: int) -> np.ndarray:
    g = grid.copy()
    g[r0:r1 + 1, c0:c1 + 1] = color
    return g


def _object_to_grid(obj: GridObject, h: int, w: int, bg: int = 0) -> np.ndarray:
    """Render object onto a grid of specified size."""
    g = np.full((h, w), bg, dtype=np.int8)
    for r, c in obj.cells:
        if 0 <= r < h and 0 <= c < w:
            g[r, c] = obj.color
    return g


def _background_color(grid: np.ndarray) -> int:
    """Detect background color (most common color)."""
    from collections import Counter
    c = Counter(int(x) for x in grid.flat)
    return c.most_common(1)[0][0]


def _render_objects(objects: list[GridObject]) -> np.ndarray:
    """
    BRIDGE PRIMITIVE: Render a list of objects back onto a grid.

    Uses the stored grid_shape from the first object to determine output
    dimensions. Objects are painted in list order (later objects on top).
    Background is 0.

    This is the critical missing link: it closes the type gap
    list[object] → grid, enabling the enumerator to build programs like:
        render_objects(extract_objects(input))
        render_objects(obj_filter_color(extract_objects(input), 3))
    """
    if not objects:
        return np.zeros((1, 1), dtype=np.int8)
    h, w = objects[0].grid_shape
    grid = np.zeros((h, w), dtype=np.int8)
    for obj in objects:
        for r, c in obj.cells:
            if 0 <= r < h and 0 <= c < w:
                grid[r, c] = obj.color
    return grid


def _render_objects_on(grid: np.ndarray, objects: list[GridObject]) -> np.ndarray:
    """
    BRIDGE PRIMITIVE: Paint objects onto a copy of the given grid.

    Unlike render_objects (which starts from a blank grid), this preserves
    the background of the input grid.  Useful for tasks where the output is
    the input with certain objects modified/recolored/moved.
    """
    result = grid.copy()
    h, w = result.shape
    for obj in objects:
        for r, c in obj.cells:
            if 0 <= r < h and 0 <= c < w:
                result[r, c] = obj.color
    return result


def _render_object(obj: GridObject) -> np.ndarray:
    """
    BRIDGE PRIMITIVE: Render a single object as a minimal bounding-box grid.

    Returns the smallest grid that contains the object (same as to_grid()
    but exposed as a concrete-typed grid→grid-compatible primitive).
    """
    if obj is None:
        return np.zeros((1, 1), dtype=np.int8)
    return obj.to_grid()


def _count_objects(objects: list[GridObject]) -> int:
    """
    BRIDGE PRIMITIVE: Count objects in a list.

    Concrete-typed alias for len() on list[object] → int.
    Unlike the polymorphic 'length' primitive, this has the concrete type
    list[object] → int, making it directly reachable by the enumerator.
    """
    return len(objects)


# ──────────────────────────────────────────────────────────────────────
# Gravity / sliding primitives
# ──────────────────────────────────────────────────────────────────────

def _gravity_down(g: np.ndarray) -> np.ndarray:
    """Slide all non-background cells downward within each column."""
    result = np.zeros_like(g)
    for c in range(g.shape[1]):
        vals = g[:, c][g[:, c] != 0]
        result[g.shape[0] - len(vals):, c] = vals
    return result


def _gravity_up(g: np.ndarray) -> np.ndarray:
    """Slide all non-background cells upward within each column."""
    result = np.zeros_like(g)
    for c in range(g.shape[1]):
        vals = g[:, c][g[:, c] != 0]
        result[:len(vals), c] = vals
    return result


def _gravity_left(g: np.ndarray) -> np.ndarray:
    """Slide all non-background cells leftward within each row."""
    result = np.zeros_like(g)
    for r in range(g.shape[0]):
        vals = g[r, :][g[r, :] != 0]
        result[r, :len(vals)] = vals
    return result


def _gravity_right(g: np.ndarray) -> np.ndarray:
    """Slide all non-background cells rightward within each row."""
    result = np.zeros_like(g)
    for r in range(g.shape[0]):
        vals = g[r, :][g[r, :] != 0]
        result[r, g.shape[1] - len(vals):] = vals
    return result


# ──────────────────────────────────────────────────────────────────────
# Flood-fill primitive
# ──────────────────────────────────────────────────────────────────────

def _flood_fill(g: np.ndarray, target_color: int, fill_color: int) -> np.ndarray:
    """
    Replace all cells of target_color with fill_color using flood fill
    (fills entire connected region reachable from any target_color cell).
    Uses 4-connectivity BFS.
    """
    if target_color == fill_color:
        return g.copy()
    result = g.copy()
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    for r in range(h):
        for c in range(w):
            if g[r, c] == target_color and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    while queue:
        r, c = queue.popleft()
        result[r, c] = fill_color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == target_color:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return result


def _fill_interior(g: np.ndarray, fill_color: int) -> np.ndarray:
    """
    Fill interior zero regions (enclosed by non-zero cells) with fill_color.
    Uses flood fill from border to identify exterior zeros, then fills the rest.
    """
    h, w = g.shape
    result = g.copy()
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    # Seed from all border zero cells
    for r in range(h):
        for c in [0, w - 1]:
            if g[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if g[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    # BFS to mark exterior zeros
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == 0:
                visited[nr, nc] = True
                queue.append((nr, nc))
    # Fill non-exterior zeros
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0 and not visited[r, c]:
                result[r, c] = fill_color
    return result


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_object_primitives(registry: PrimitiveRegistry) -> None:
    """Register all L_objects primitives."""
    S = SpelkeSystem.OBJECTS
    props = ["domain-specific", "task-specific", "abstract", "encapsulated"]

    primitives = [
        Primitive("extract_objects", Arrow(tgrid, tlist(tobject)),
                  lambda g: _extract_objects(g), S,
                  "Cohesion: extract connected components (4-connected)", props),
        Primitive("extract_objects_8conn", Arrow(tgrid, tlist(tobject)),
                  lambda g: _extract_objects(g, connectivity=8), S,
                  "Cohesion: extract connected components (8-connected)", props),
        Primitive("extract_objects_multicolor", Arrow(tgrid, tlist(tobject)),
                  lambda g: _extract_objects_multicolor(g), S,
                  "Extract objects ignoring color boundaries", props),
        Primitive("obj_touching", Arrow(tobject, Arrow(tobject, tbool)),
                  _touching, S, "Contact: are two objects 4-adjacent?", props),
        Primitive("obj_overlapping", Arrow(tobject, Arrow(tobject, tbool)),
                  _overlapping, S, "Do two objects share cells?", props),
        Primitive("obj_z_order", Arrow(tlist(tobject), tlist(tobject)),
                  _z_order, S, "Occlusion: order by implied depth", props),
        Primitive("obj_copy", Arrow(tobject, tobject),
                  _copy_object, S, "Copy an object", props),
        Primitive("obj_translate", Arrow(tobject, Arrow(tint, Arrow(tint, tobject))),
                  lambda o: lambda dr: lambda dc: _translate_object(o, dr, dc), S,
                  "Continuity: translate by (dr, dc)", props),
        Primitive("obj_rotate90", Arrow(tobject, tobject),
                  _rotate_object_90, S, "Rotate 90° clockwise", props),
        Primitive("obj_reflect_h", Arrow(tobject, tobject),
                  _reflect_object_h, S, "Reflect horizontally", props),
        Primitive("obj_reflect_v", Arrow(tobject, tobject),
                  _reflect_object_v, S, "Reflect vertically", props),
        Primitive("obj_bbox", Arrow(tobject, tpair(tpoint, tpoint)),
                  lambda o: _bbox(o), S, "Bounding box", props),
        Primitive("obj_recolor", Arrow(tobject, Arrow(tcolor, tobject)),
                  lambda o: lambda c: _recolor_object(o, c), S,
                  "Change object color", props),
        Primitive("obj_color", Arrow(tobject, tcolor),
                  _object_color, S, "Get object color", props),
        Primitive("obj_size", Arrow(tobject, tint),
                  _object_size, S, "Get object cell count", props),
        Primitive("obj_position", Arrow(tobject, tpoint),
                  _object_position, S, "Get top-left position", props),
        Primitive("obj_same_shape", Arrow(tobject, Arrow(tobject, tbool)),
                  _objects_same_shape, S,
                  "Persistence: translation-invariant shape match", props),
        Primitive("obj_filter_color", Arrow(tlist(tobject), Arrow(tcolor, tlist(tobject))),
                  lambda objs: lambda c: _filter_by_color(objs, c), S,
                  "Filter objects by color", props),
        Primitive("obj_filter_size", Arrow(tlist(tobject), Arrow(tint, tlist(tobject))),
                  lambda objs: lambda s: _filter_by_size(objs, s), S,
                  "Filter objects by size", props),
        Primitive("obj_largest", Arrow(tlist(tobject), tobject),
                  _largest_object, S, "Get largest object", props),
        Primitive("obj_smallest", Arrow(tlist(tobject), tobject),
                  _smallest_object, S, "Get smallest object", props),
        Primitive("obj_sort_size", Arrow(tlist(tobject), tlist(tobject)),
                  _sort_objects_by_size, S, "Sort by size ascending", props),
        Primitive("obj_sort_position", Arrow(tlist(tobject), tlist(tobject)),
                  _sort_objects_by_position, S, "Sort by position (top-left)", props),
        Primitive("obj_to_grid", Arrow(tobject, Arrow(tint, Arrow(tint, tgrid))),
                  lambda o: lambda h: lambda w: _object_to_grid(o, h, w), S,
                  "Render object to grid of size h×w", props),
        Primitive("crop", Arrow(tgrid, Arrow(tint, Arrow(tint, Arrow(tint, Arrow(tint, tgrid))))),
                  lambda g: lambda r0: lambda c0: lambda r1: lambda c1: _crop_grid(g, r0, c0, r1, c1), S,
                  "Crop grid region", props),
        Primitive("fill_rect", Arrow(tgrid, Arrow(tint, Arrow(tint, Arrow(tint, Arrow(tint, Arrow(tcolor, tgrid)))))),
                  lambda g: lambda r0: lambda c0: lambda r1: lambda c1: lambda col: _fill_rect(g, r0, c0, r1, c1, col), S,
                  "Fill rectangular region with color", props),
        Primitive("grid_colors", Arrow(tgrid, tlist(tcolor)),
                  _grid_colors, S, "Get unique non-background colors", props),
        Primitive("background_color", Arrow(tgrid, tcolor),
                  _background_color, S, "Detect background (most common) color", props),

        # ── Bridge primitives: close the list[object] → grid type gap ──
        Primitive("render_objects", Arrow(tlist(tobject), tgrid),
                  _render_objects, S,
                  "Bridge: render list of objects back onto grid-shaped canvas", props),
        Primitive("render_objects_on", Arrow(tgrid, Arrow(tlist(tobject), tgrid)),
                  lambda g: lambda objs: _render_objects_on(g, objs), S,
                  "Bridge: paint objects onto existing grid (preserves background)", props),
        Primitive("render_object", Arrow(tobject, tgrid),
                  _render_object, S,
                  "Bridge: render single object as minimal bounding-box grid", props),
        Primitive("count_objects", Arrow(tlist(tobject), tint),
                  _count_objects, S,
                  "Bridge: count objects in list (concrete-typed)", props),

        # ── Gravity / sliding primitives ──
        Primitive("gravity_down", Arrow(tgrid, tgrid),
                  _gravity_down, S,
                  "Slide non-background cells downward within each column", props),
        Primitive("gravity_up", Arrow(tgrid, tgrid),
                  _gravity_up, S,
                  "Slide non-background cells upward within each column", props),
        Primitive("gravity_left", Arrow(tgrid, tgrid),
                  _gravity_left, S,
                  "Slide non-background cells leftward within each row", props),
        Primitive("gravity_right", Arrow(tgrid, tgrid),
                  _gravity_right, S,
                  "Slide non-background cells rightward within each row", props),

        # ── Flood-fill / interior-fill primitives ──
        Primitive("flood_fill", Arrow(tgrid, Arrow(tcolor, Arrow(tcolor, tgrid))),
                  lambda g: lambda tc: lambda fc: _flood_fill(g, tc, fc), S,
                  "Replace connected region of target_color with fill_color", props),
        Primitive("fill_interior", Arrow(tgrid, Arrow(tcolor, tgrid)),
                  lambda g: lambda fc: _fill_interior(g, fc), S,
                  "Fill enclosed interior zero regions with fill_color", props),
    ]

    for p in primitives:
        registry.register(p)
