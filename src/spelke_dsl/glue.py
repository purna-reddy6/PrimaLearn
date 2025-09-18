"""
glue.py — Generic combinators + NSM logical/causal operators.

These are NOT Spelke core systems. They are:
1. Standard functional combinators (compose, map, filter, fold, conditional, lambda)
2. NSM-derived logical/causal operators (Wierzbicka & Goddard 2014)

The NSM operators are precisely what Spelke's six systems lack:
the modeling-process operators that Carey-style bootstrapping requires
(BECAUSE, IF, NOT, SAY, TRUE, MAYBE, CAN).

Section 7.2 of the Master Plan: "Spelke for what, NSM for how."
"""

from __future__ import annotations
from typing import Any, Callable
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Arrow,
    tgrid, tobject, tcolor, tint, tbool, tlist, tpair,
    t0, t1, t2, TypeVariable,
)


# ──────────────────────────────────────────────────────────────────────
# Functional combinators
# ──────────────────────────────────────────────────────────────────────

def _compose(f: Callable, g: Callable) -> Callable:
    """compose f g x = f(g(x))"""
    return lambda x: f(g(x))

def _map_fn(f: Callable, items: list) -> list:
    return [f(x) for x in items]

def _filter_fn(pred: Callable, items: list) -> list:
    return [x for x in items if pred(x)]

def _fold_left(f: Callable, init: Any, items: list) -> Any:
    acc = init
    for x in items:
        acc = f(acc)(x)
    return acc

def _if_then_else(cond: bool, then_val: Any, else_val: Any) -> Any:
    return then_val if cond else else_val

def _identity(x: Any) -> Any:
    return x

def _const(x: Any) -> Callable:
    return lambda _: x

def _pair(a: Any, b: Any) -> tuple:
    return (a, b)

def _fst(p: tuple) -> Any:
    return p[0]

def _snd(p: tuple) -> Any:
    return p[1]

def _cons(x: Any, xs: list) -> list:
    return [x] + xs

def _head(xs: list) -> Any:
    return xs[0] if xs else None

def _tail(xs: list) -> list:
    return xs[1:] if xs else []

def _empty(xs: list) -> bool:
    return len(xs) == 0

def _singleton(x: Any) -> list:
    return [x]

def _concat(xs: list, ys: list) -> list:
    return xs + ys

def _reverse(xs: list) -> list:
    return list(reversed(xs))

def _zip_lists(xs: list, ys: list) -> list[tuple]:
    return list(zip(xs, ys))

def _enumerate_list(xs: list) -> list[tuple]:
    return list(enumerate(xs))

def _flatten(xss: list[list]) -> list:
    result = []
    for xs in xss:
        result.extend(xs)
    return result

def _take(n: int, xs: list) -> list:
    return xs[:n]

def _drop(n: int, xs: list) -> list:
    return xs[n:]

def _any_pred(pred: Callable, items: list) -> bool:
    return any(pred(x) for x in items)

def _all_pred(pred: Callable, items: list) -> bool:
    return all(pred(x) for x in items)

def _find(pred: Callable, items: list) -> Any:
    for x in items:
        if pred(x):
            return x
    return None


# ──────────────────────────────────────────────────────────────────────
# NSM logical/causal/communicative operators
# ──────────────────────────────────────────────────────────────────────

def _not(x: bool) -> bool:
    """NSM: NOT"""
    return not x

def _and(a: bool, b: bool) -> bool:
    return a and b

def _or(a: bool, b: bool) -> bool:
    return a or b

def _same(a: Any, b: Any) -> bool:
    """NSM: THE SAME — deep equality."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_same(x, y) for x, y in zip(a, b))
    try:
        import numpy as np
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
    except ImportError:
        pass
    return a == b

def _other(items: list, excluded: Any) -> list:
    """NSM: OTHER — everything except the excluded item."""
    return [x for x in items if not _same(x, excluded)]


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────

def register_glue_primitives(registry: PrimitiveRegistry) -> None:
    """Register generic combinators and NSM operators."""
    G = SpelkeSystem.GLUE
    N = SpelkeSystem.NSM

    glue_prims = [
        # Higher-order combinators
        Primitive("map", Arrow(Arrow(t0, t1), Arrow(tlist(t0), tlist(t1))),
                  lambda f: lambda xs: _map_fn(f, xs), G,
                  "Map function over list"),
        Primitive("filter", Arrow(Arrow(t0, tbool), Arrow(tlist(t0), tlist(t0))),
                  lambda p: lambda xs: _filter_fn(p, xs), G,
                  "Filter list by predicate"),
        Primitive("fold", Arrow(Arrow(t0, Arrow(t1, t0)), Arrow(t0, Arrow(tlist(t1), t0))),
                  lambda f: lambda init: lambda xs: _fold_left(f, init, xs), G,
                  "Left fold"),
        Primitive("if", Arrow(tbool, Arrow(t0, Arrow(t0, t0))),
                  lambda c: lambda t: lambda e: _if_then_else(c, t, e), G,
                  "Conditional"),
        Primitive("compose", Arrow(Arrow(t1, t2), Arrow(Arrow(t0, t1), Arrow(t0, t2))),
                  lambda f: lambda g: lambda x: f(g(x)), G,
                  "Function composition"),
        Primitive("identity", Arrow(t0, t0), _identity, G, "Identity function"),
        Primitive("const", Arrow(t0, Arrow(t1, t0)),
                  lambda x: lambda _: x, G, "Constant function"),

        # Pair operations
        Primitive("pair", Arrow(t0, Arrow(t1, tpair(t0, t1))),
                  lambda a: lambda b: (a, b), G, "Construct pair"),
        Primitive("fst", Arrow(tpair(t0, t1), t0), _fst, G, "First of pair"),
        Primitive("snd", Arrow(tpair(t0, t1), t1), _snd, G, "Second of pair"),

        # List operations
        Primitive("cons", Arrow(t0, Arrow(tlist(t0), tlist(t0))),
                  lambda x: lambda xs: _cons(x, xs), G, "Prepend to list"),
        Primitive("head", Arrow(tlist(t0), t0), _head, G, "First element"),
        Primitive("tail", Arrow(tlist(t0), tlist(t0)), _tail, G, "All but first"),
        Primitive("empty", Arrow(tlist(t0), tbool), _empty, G, "Is list empty?"),
        Primitive("singleton", Arrow(t0, tlist(t0)), _singleton, G, "Wrap in list"),
        Primitive("concat", Arrow(tlist(t0), Arrow(tlist(t0), tlist(t0))),
                  lambda a: lambda b: _concat(a, b), G, "Concatenate lists"),
        Primitive("reverse", Arrow(tlist(t0), tlist(t0)), _reverse, G, "Reverse list"),
        Primitive("flatten", Arrow(tlist(tlist(t0)), tlist(t0)), _flatten, G,
                  "Flatten nested list"),
        Primitive("take", Arrow(tint, Arrow(tlist(t0), tlist(t0))),
                  lambda n: lambda xs: _take(n, xs), G, "Take first n"),
        Primitive("drop", Arrow(tint, Arrow(tlist(t0), tlist(t0))),
                  lambda n: lambda xs: _drop(n, xs), G, "Drop first n"),
        Primitive("any", Arrow(Arrow(t0, tbool), Arrow(tlist(t0), tbool)),
                  lambda p: lambda xs: _any_pred(p, xs), G, "Any element satisfies?"),
        Primitive("all", Arrow(Arrow(t0, tbool), Arrow(tlist(t0), tbool)),
                  lambda p: lambda xs: _all_pred(p, xs), G, "All elements satisfy?"),
        Primitive("find", Arrow(Arrow(t0, tbool), Arrow(tlist(t0), t0)),
                  lambda p: lambda xs: _find(p, xs), G,
                  "Find first matching element"),
    ]

    nsm_prims = [
        # NSM logical
        Primitive("not", Arrow(tbool, tbool), _not, N, "NSM: NOT"),
        Primitive("and", Arrow(tbool, Arrow(tbool, tbool)), _and, N, "Logical AND"),
        Primitive("or", Arrow(tbool, Arrow(tbool, tbool)), _or, N, "Logical OR"),
        Primitive("true", tbool, lambda: True, N, "NSM: TRUE"),
        Primitive("false", tbool, lambda: False, N, "NSM: FALSE"),
        Primitive("same", Arrow(t0, Arrow(t0, tbool)),
                  _same, N, "NSM: THE SAME — deep equality"),
        Primitive("other", Arrow(tlist(t0), Arrow(t0, tlist(t0))),
                  lambda items: lambda ex: _other(items, ex), N,
                  "NSM: OTHER — all except excluded"),
    ]

    for p in glue_prims + nsm_prims:
        registry.register(p)
