"""
generic_dsl.py — Matched-cardinality generic λ-calculus DSL.

THE CRITICAL CONTROL: same number of primitives as Spelke DSL,
but generic functional combinators instead of Spelke-organized modules.

This isolates "right primitives" from "more primitives."
If Spelke-initialized LILO outperforms this baseline, the gain comes
from the *content* of the priors, not their *count*.
"""

from __future__ import annotations
import numpy as np
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tpoint, tlist, tpair,
    t0, t1,
)
from src.engine.library import Library


def build_generic_dsl(target_cardinality: int | None = None) -> PrimitiveRegistry:
    """
    Build a generic λ-calculus DSL with matched cardinality.

    If target_cardinality is provided, pad or trim to match
    the Spelke DSL's primitive count exactly.
    """
    registry = PrimitiveRegistry()
    S = SpelkeSystem.GLUE

    # ── Standard functional combinators ──
    generics = [
        # List operations
        Primitive("g_map", Arrow(Arrow(t0, t1), Arrow(tlist(t0), tlist(t1))),
                  lambda f: lambda xs: [f(x) for x in xs], S, "map"),
        Primitive("g_filter", Arrow(Arrow(t0, tbool), Arrow(tlist(t0), tlist(t0))),
                  lambda p: lambda xs: [x for x in xs if p(x)], S, "filter"),
        Primitive("g_fold", Arrow(Arrow(t0, Arrow(t1, t0)), Arrow(t0, Arrow(tlist(t1), t0))),
                  lambda f: lambda init: lambda xs: _fold(f, init, xs), S, "fold"),
        Primitive("g_cons", Arrow(t0, Arrow(tlist(t0), tlist(t0))),
                  lambda x: lambda xs: [x] + xs, S, "cons"),
        Primitive("g_head", Arrow(tlist(t0), t0),
                  lambda xs: xs[0] if xs else None, S, "head"),
        Primitive("g_tail", Arrow(tlist(t0), tlist(t0)),
                  lambda xs: xs[1:] if xs else [], S, "tail"),
        Primitive("g_length", Arrow(tlist(t0), tint),
                  lambda xs: len(xs), S, "length"),
        Primitive("g_concat", Arrow(tlist(t0), Arrow(tlist(t0), tlist(t0))),
                  lambda a: lambda b: a + b, S, "concat"),
        Primitive("g_reverse", Arrow(tlist(t0), tlist(t0)),
                  lambda xs: list(reversed(xs)), S, "reverse"),
        Primitive("g_empty", Arrow(tlist(t0), tbool),
                  lambda xs: len(xs) == 0, S, "empty?"),

        # Arithmetic
        Primitive("g_add", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: a + b, S, "add"),
        Primitive("g_sub", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: max(0, a - b), S, "sub"),
        Primitive("g_mul", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: a * b, S, "mul"),
        Primitive("g_div", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: a // b if b else 0, S, "div"),
        Primitive("g_mod", Arrow(tint, Arrow(tint, tint)),
                  lambda a: lambda b: a % b if b else 0, S, "mod"),
        Primitive("g_eq", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: a == b, S, "eq"),
        Primitive("g_gt", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: a > b, S, "gt"),
        Primitive("g_lt", Arrow(tint, Arrow(tint, tbool)),
                  lambda a: lambda b: a < b, S, "lt"),

        # Boolean
        Primitive("g_not", Arrow(tbool, tbool), lambda x: not x, S, "not"),
        Primitive("g_and", Arrow(tbool, Arrow(tbool, tbool)),
                  lambda a: lambda b: a and b, S, "and"),
        Primitive("g_or", Arrow(tbool, Arrow(tbool, tbool)),
                  lambda a: lambda b: a or b, S, "or"),
        Primitive("g_if", Arrow(tbool, Arrow(t0, Arrow(t0, t0))),
                  lambda c: lambda t: lambda e: t if c else e, S, "if"),

        # Higher-order
        Primitive("g_compose", Arrow(Arrow(t1, t0), Arrow(Arrow(t0, t1), Arrow(t0, t0))),
                  lambda f: lambda g: lambda x: f(g(x)), S, "compose"),
        Primitive("g_identity", Arrow(t0, t0), lambda x: x, S, "identity"),
        Primitive("g_const", Arrow(t0, Arrow(t1, t0)),
                  lambda x: lambda _: x, S, "const"),

        # Pairs
        Primitive("g_pair", Arrow(t0, Arrow(t1, tpair(t0, t1))),
                  lambda a: lambda b: (a, b), S, "pair"),
        Primitive("g_fst", Arrow(tpair(t0, t1), t0),
                  lambda p: p[0], S, "fst"),
        Primitive("g_snd", Arrow(tpair(t0, t1), t1),
                  lambda p: p[1], S, "snd"),

        # Grid operations (generic, not Spelke-organized)
        Primitive("g_get_cell", Arrow(tgrid, Arrow(tint, Arrow(tint, tcolor))),
                  lambda g: lambda r: lambda c: int(g[r, c]) if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] else 0,
                  S, "get cell"),
        Primitive("g_set_cell", Arrow(tgrid, Arrow(tint, Arrow(tint, Arrow(tcolor, tgrid)))),
                  lambda g: lambda r: lambda c: lambda v: _set_cell(g, r, c, v),
                  S, "set cell"),
        Primitive("g_height", Arrow(tgrid, tint), lambda g: g.shape[0], S, "height"),
        Primitive("g_width", Arrow(tgrid, tint), lambda g: g.shape[1], S, "width"),
        Primitive("g_rotate", Arrow(tgrid, tgrid),
                  lambda g: np.rot90(g, k=-1).copy(), S, "rotate90"),
        Primitive("g_flip", Arrow(tgrid, tgrid),
                  lambda g: np.fliplr(g).copy(), S, "flip"),
        Primitive("g_transpose", Arrow(tgrid, tgrid),
                  lambda g: g.T.copy(), S, "transpose"),

        # Constants
        Primitive("g_0", tint, lambda: 0, S, "0"),
        Primitive("g_1", tint, lambda: 1, S, "1"),
        Primitive("g_2", tint, lambda: 2, S, "2"),
        Primitive("g_3", tint, lambda: 3, S, "3"),
        Primitive("g_4", tint, lambda: 4, S, "4"),
        Primitive("g_5", tint, lambda: 5, S, "5"),
        Primitive("g_6", tint, lambda: 6, S, "6"),
        Primitive("g_7", tint, lambda: 7, S, "7"),
        Primitive("g_8", tint, lambda: 8, S, "8"),
        Primitive("g_9", tint, lambda: 9, S, "9"),
        Primitive("g_true", tbool, lambda: True, S, "true"),
        Primitive("g_false", tbool, lambda: False, S, "false"),
    ]

    for p in generics:
        registry.register(p)

    # Pad to target cardinality (default: match Spelke DSL at 131)
    if target_cardinality is None:
        target_cardinality = 131
    
    current = len(registry._primitives)
    pad_idx = 0
    while current < target_cardinality:
        # Add grid transform variants as padding
        pad_ops = [
            (f"g_rot{pad_idx}", Arrow(tgrid, tgrid),
             lambda g, k=pad_idx % 3 + 1: np.rot90(g, k=k).copy(), f"rotate variant {pad_idx}"),
            (f"g_neg{pad_idx}", Arrow(tint, tint),
             lambda x, k=pad_idx + 2: x * k, f"multiply by {pad_idx+2}"),
            (f"g_clamp{pad_idx}", Arrow(tint, tint),
             lambda x, lo=pad_idx % 5, hi=5 + pad_idx % 5: max(lo, min(hi, x)),
             f"clamp({pad_idx%5},{5+pad_idx%5})"),
        ]
        for name, typ, impl, desc in pad_ops:
            if current >= target_cardinality:
                break
            registry.register(Primitive(name, typ, impl, S, desc))
            current += 1
        pad_idx += 1

    return registry


def _fold(f, init, xs):
    acc = init
    for x in xs:
        acc = f(acc)(x)
    return acc


def _set_cell(g, r, c, v):
    result = g.copy()
    if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
        result[r, c] = v
    return result


def build_generic_library(target_cardinality: int | None = None) -> Library:
    """Build a Library from the generic DSL."""
    registry = build_generic_dsl(target_cardinality)
    return Library(registry)
