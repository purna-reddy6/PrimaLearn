"""
l_number.py — Core System 6: Number

Two empirically distinct subsystems (Spelke 2000, Spelke & Kinzler 2007):
1. Parallel Individuation / Object Tracking System (OTS): exact, cap ≤3
   (Feigenson & Carey 2003; Wynn 1992 addition/subtraction)
2. Approximate Number System (ANS): ratio-dependent magnitude estimation
   (Halberda et al.; Xu & Spelke 2000 large-number discrimination at 6 months)

The cardinal-principle bootstrap (ages 2½–4) integrates these two systems
plus the count-list placeholder — this is the canonical Carey case.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Any
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tlist, tpair, tfloat,
)


# ──────────────────────────────────────────────────────────────────────
# Parallel Individuation (OTS) — exact, ≤3
# ──────────────────────────────────────────────────────────────────────

def _count_exact_small(items: list) -> int:
    """
    OTS: exact count, but only reliable for ≤3.
    Returns exact count for ≤3, -1 for >3 (signature limit).
    This models the empirical cap of the parallel individuation system.
    """
    n = len(items)
    if n <= 3:
        return n
    return -1  # "many" — beyond OTS capacity


def _is_one(items: list) -> bool:
    return len(items) == 1

def _is_two(items: list) -> bool:
    return len(items) == 2

def _is_three(items: list) -> bool:
    return len(items) == 3

def _ots_equal(a: list, b: list) -> bool:
    """OTS comparison: works only for small sets."""
    if len(a) > 3 or len(b) > 3:
        return False  # Cannot determine
    return len(a) == len(b)


# ──────────────────────────────────────────────────────────────────────
# Approximate Number System (ANS) — ratio-dependent
# ──────────────────────────────────────────────────────────────────────

ANS_WEBER_FRACTION = 0.15  # Typical adult Weber fraction for number

def _ans_greater(n1: int, n2: int) -> bool:
    """
    ANS comparison: reliable only when ratio is sufficiently distinct.
    Weber's law: discriminability depends on n1/n2 ratio.
    Returns True if n1 is detectably greater than n2.
    """
    if n2 == 0:
        return n1 > 0
    if n1 == 0:
        return False
    ratio = max(n1, n2) / min(n1, n2)
    # Discriminable if ratio exceeds 1 + Weber fraction
    if ratio > 1 + ANS_WEBER_FRACTION:
        return n1 > n2
    return False  # Too close to distinguish

def _ans_approximate_equal(n1: int, n2: int) -> bool:
    """ANS: are two quantities approximately equal?"""
    if n1 == 0 and n2 == 0:
        return True
    if n1 == 0 or n2 == 0:
        return False
    ratio = max(n1, n2) / min(n1, n2)
    return ratio <= 1 + ANS_WEBER_FRACTION


# ──────────────────────────────────────────────────────────────────────
# Exact arithmetic (bootstrapped above OTS via count-list placeholder)
# ──────────────────────────────────────────────────────────────────────

def _count(items: list) -> int:
    """Full exact count — the bootstrapped cardinal principle."""
    return len(items)

def _add(a: int, b: int) -> int:
    return a + b

def _sub(a: int, b: int) -> int:
    return max(0, a - b)

def _mul(a: int, b: int) -> int:
    return a * b

def _div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return a // b

def _mod(a: int, b: int) -> int:
    if b == 0:
        return 0
    return a % b

def _eq(a: int, b: int) -> bool:
    return a == b

def _neq(a: int, b: int) -> bool:
    return a != b

def _gt(a: int, b: int) -> bool:
    return a > b

def _lt(a: int, b: int) -> bool:
    return a < b

def _gte(a: int, b: int) -> bool:
    return a >= b

def _lte(a: int, b: int) -> bool:
    return a <= b

def _max_val(a: int, b: int) -> int:
    return max(a, b)

def _min_val(a: int, b: int) -> int:
    return min(a, b)

def _abs_val(a: int) -> int:
    return abs(a)

def _successor(n: int) -> int:
    """The successor function — the formal heart of the cardinal-principle bootstrap."""
    return n + 1

def _predecessor(n: int) -> int:
    return max(0, n - 1)

def _range_list(n: int) -> list[int]:
    """Generate [0, 1, ..., n-1]."""
    return list(range(n))


# ──────────────────────────────────────────────────────────────────────
# Set/collection operations on numbers
# ──────────────────────────────────────────────────────────────────────

def _sum_list(items: list[int]) -> int:
    return sum(items)

def _max_list(items: list[int]) -> int:
    if not items:
        return 0
    return max(items)

def _min_list(items: list[int]) -> int:
    if not items:
        return 0
    return min(items)

def _sort_ascending(items: list[int]) -> list[int]:
    return sorted(items)

def _sort_descending(items: list[int]) -> list[int]:
    return sorted(items, reverse=True)

def _unique(items: list[int]) -> list[int]:
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

def _length(items: list) -> int:
    return len(items)

def _nth(items: list, n: int) -> Any:
    if 0 <= n < len(items):
        return items[n]
    return None

def _first(items: list) -> Any:
    return items[0] if items else None

def _last(items: list) -> Any:
    return items[-1] if items else None


# ──────────────────────────────────────────────────────────────────────
# NUMBER+OBJECTS cross-system bridge implementations
# Registered under SpelkeSystem.NUMBER — NUMBER logic drives these.
# ──────────────────────────────────────────────────────────────────────

def _render_count_colored(n: int, color: int) -> np.ndarray:
    """
    BRIDGE PRIMITIVE (NUMBER+OBJECTS crossing): Render a count as a 1×N row.

    Takes an integer count n and a color, returns a 1×n numpy grid where
    every cell is filled with `color`. This is the canonical NUMBER→OBJECTS
    bridge: the NUMBER system produces a count, OBJECTS renders it visually.

    Registered under SpelkeSystem.NUMBER so Stitch sees NUMBER in
    systems_composed when this primitive appears in an abstraction.

    Args:
        n: Number of cells to render (count). If ≤ 0, returns 1×1 zeros.
        color: The color to fill each cell with.

    Returns:
        np.ndarray of shape (1, n) filled with `color`, dtype int8.
    """
    if n <= 0:
        return np.zeros((1, 1), dtype=np.int8)
    row = np.full((1, n), int(color), dtype=np.int8)
    return row


def _tile_n(grid: np.ndarray, n: int) -> np.ndarray:
    """
    BRIDGE PRIMITIVE (NUMBER+OBJECTS crossing): Tile a grid N times horizontally.

    Takes a grid and an integer n, returns a new grid with the original
    repeated n times side by side (horizontal concatenation). This is the
    canonical NUMBER+OBJECTS replicate bridge: NUMBER system provides the
    count, OBJECTS system provides the extracted shape to replicate.

    Registered under SpelkeSystem.NUMBER so Stitch sees NUMBER in
    systems_composed when this primitive appears in an abstraction.

    Args:
        grid: The source grid to replicate.
        n: Number of times to tile. If ≤ 0 or 1, returns grid as-is.

    Returns:
        np.ndarray with n copies of grid side by side.
    """
    if n <= 1:
        return grid.copy()
    tiles = [grid] * int(n)
    return np.hstack(tiles).astype(grid.dtype)


def _count_in_quadrant(grid: np.ndarray, quadrant: int) -> int:
    """
    BRIDGE PRIMITIVE (NUMBER+PLACES crossing): Count objects in a specific quadrant.

    Extracts the specified quadrant (0=TL, 1=TR, 2=BL, 3=BR) then counts
    distinct connected components in it. Bridges PLACES (spatial layout) and
    NUMBER (counting). Used in Type 5 tasks.
    """
    from src.spelke_dsl.l_places import _extract_quadrant
    from src.spelke_dsl.l_objects import _extract_objects
    quad = _extract_quadrant(grid, quadrant)
    return len(_extract_objects(quad))


def _count_cells(grid: np.ndarray) -> int:
    """
    BRIDGE PRIMITIVE (NUMBER+OBJECTS crossing): Count total non-zero cells in grid.

    Unlike count_objects (which counts connected components / distinct objects),
    this counts every individual non-zero cell. This bridges OBJECTS (the grid
    representation) to NUMBER (the total cell count).

    Key distinction from count_objects:
    - count_objects([##, ##]) = 1 (one connected region)
    - count_cells([##, ##])   = 4 (four non-zero cells)

    This enables solving ARC task d631b094: input has N scattered cells of a
    color, output is a 1×N row of that color. The program is:
      render_count_colored(count_cells(input), obj_color(obj_largest(...)))

    Registered under SpelkeSystem.NUMBER so Stitch sees NUMBER in
    systems_composed → is_cross_system = True for abstractions using it.

    Args:
        grid: A 2D numpy array (ARC grid).

    Returns:
        int: Number of non-zero cells.
    """
    return int(np.count_nonzero(grid))


# ──────────────────────────────────────────────────────────────────────
# Constants (small numbers — within OTS range + common ARC values)
# ──────────────────────────────────────────────────────────────────────

# These are registered as 0-arity primitives (constants)


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_number_primitives(registry: PrimitiveRegistry) -> None:
    """Register all L_number primitives."""
    S = SpelkeSystem.NUMBER
    props = ["domain-specific", "abstract"]

    primitives = [
        # OTS — parallel individuation (cap 3)
        Primitive("ots_count", Arrow(tlist(tobject), tint),
                  _count_exact_small, S,
                  "OTS: exact count ≤3, returns -1 for >3 (signature limit)", props),
        Primitive("is_one", Arrow(tlist(tobject), tbool), _is_one, S,
                  "OTS: exactly one item?", props),
        Primitive("is_two", Arrow(tlist(tobject), tbool), _is_two, S,
                  "OTS: exactly two items?", props),
        Primitive("is_three", Arrow(tlist(tobject), tbool), _is_three, S,
                  "OTS: exactly three items?", props),

        # ANS — approximate magnitude
        Primitive("ans_greater", Arrow(tint, Arrow(tint, tbool)),
                  _ans_greater, S,
                  "ANS: is n1 detectably greater than n2? (Weber's law)", props),
        Primitive("ans_approx_equal", Arrow(tint, Arrow(tint, tbool)),
                  _ans_approximate_equal, S,
                  "ANS: are n1 and n2 approximately equal?", props),

        # Bootstrapped exact arithmetic
        Primitive("count", Arrow(tlist(tobject), tint), _count, S,
                  "Cardinal principle: exact count (bootstrapped)", props),
        Primitive("add", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: _add(a, b), S, "Addition", props),
        Primitive("sub", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: _sub(a, b), S, "Subtraction (floored at 0)", props),
        Primitive("mul", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: _mul(a, b), S, "Multiplication", props),
        Primitive("div", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: _div(a, b), S, "Integer division", props),
        Primitive("mod", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: _mod(a, b), S, "Modulo", props),
        Primitive("eq", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: _eq(a, b), S, "Equality", props),
        Primitive("neq", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: _neq(a, b), S, "Inequality", props),
        Primitive("gt", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: _gt(a, b), S, "Greater than", props),
        Primitive("lt", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: _lt(a, b), S, "Less than", props),
        Primitive("gte", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: _gte(a, b), S, "Greater or equal", props),
        Primitive("lte", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: _lte(a, b), S, "Less or equal", props),
        Primitive("max2", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: _max_val(a, b), S, "Maximum of two", props),
        Primitive("min2", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: _min_val(a, b), S, "Minimum of two", props),
        Primitive("abs", Arrow(tint, tint), _abs_val, S, "Absolute value", props),
        Primitive("succ", Arrow(tint, tint), _successor, S,
                  "Successor function — heart of cardinal-principle bootstrap", props),
        Primitive("pred", Arrow(tint, tint), _predecessor, S,
                  "Predecessor (floored at 0)", props),
        Primitive("range", Arrow(tint, tlist(tint)), _range_list, S,
                  "Generate [0..n-1]", props),

        # Collection operations
        Primitive("sum", Arrow(tlist(tint), tint), _sum_list, S, "Sum of list", props),
        Primitive("max_list", Arrow(tlist(tint), tint), _max_list, S, "Max of list", props),
        Primitive("min_list", Arrow(tlist(tint), tint), _min_list, S, "Min of list", props),
        Primitive("sort_asc", Arrow(tlist(tint), tlist(tint)),
                  _sort_ascending, S, "Sort ascending", props),
        Primitive("sort_desc", Arrow(tlist(tint), tlist(tint)),
                  _sort_descending, S, "Sort descending", props),
        Primitive("unique", Arrow(tlist(tint), tlist(tint)),
                  _unique, S, "Unique elements preserving order", props),
        Primitive("length", Arrow(tlist(tobject), tint), _length, S,
                  "Length of list", props),

        # ── NUMBER+OBJECTS cross-system bridge primitives (NUMBER side) ──
        # These are NUMBER system primitives: NUMBER produces the count/tiling
        # logic; OBJECTS system provides the grid/object inputs.
        # Registering under SpelkeSystem.NUMBER so Stitch sees NUMBER in
        # systems_composed → is_cross_system = True.
        Primitive("render_count_colored", Arrow(tint, Arrow(tcolor, tgrid)),
                  lambda n: lambda c: _render_count_colored(n, c), S,
                  "NUMBER+OBJECTS bridge: render count n as 1×n grid of given color", props),
        Primitive("tile_n", Arrow(tgrid, Arrow(tint, tgrid)),
                  lambda g: lambda n: _tile_n(g, n), S,
                  "NUMBER+OBJECTS bridge: tile grid n times horizontally", props),
        Primitive("count_cells", Arrow(tgrid, tint),
                  _count_cells, S,
                  "NUMBER+OBJECTS bridge: count total non-zero cells in grid (not connected components)", props),
        Primitive("count_in_quadrant", Arrow(tgrid, Arrow(tint, tint)),
                  lambda g: lambda q: _count_in_quadrant(g, q), S,
                  "NUMBER+PLACES bridge: count objects in given quadrant (0=TL,1=TR,2=BL,3=BR)", props),

        # Constants (0-9 for ARC color palette)
        Primitive("zero", tint, lambda: 0, S, "Constant 0", props),
        Primitive("one", tint, lambda: 1, S, "Constant 1", props),
        Primitive("two", tint, lambda: 2, S, "Constant 2", props),
        Primitive("three", tint, lambda: 3, S, "Constant 3", props),
        Primitive("four", tint, lambda: 4, S, "Constant 4", props),
        Primitive("five", tint, lambda: 5, S, "Constant 5", props),
        Primitive("six", tint, lambda: 6, S, "Constant 6", props),
        Primitive("seven", tint, lambda: 7, S, "Constant 7", props),
        Primitive("eight", tint, lambda: 8, S, "Constant 8", props),
        Primitive("nine", tint, lambda: 9, S, "Constant 9", props),
    ]

    for p in primitives:
        registry.register(p)
