"""
stitch.py — Anti-unification based compression (Stitch-style).

Replaces simple subtree matching with proper anti-unification:
discovers abstractions with HOLES like λx. λy. (f x (g y))
instead of only exact subtree matches.

New in this version:
  - rewrite_corpus(): rewrites the program corpus to use discovered abstractions
  - register_abstractions_as_primitives(): registers abstractions as real callable
    Primitives in the registry so the enumerator can actually invoke them.

Based on Bowers et al., POPL 2023.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Optional
from src.engine.program import (
    Program, ProgramNode, PrimNode, AppNode, LamNode, VarNode, LitNode,
)
from src.engine.library import Library, Abstraction
from src.spelke_dsl.base import Type, Arrow, TypeVariable


class AntiUnifier:
    """
    Anti-unification: find the most specific generalization of two terms.

    Given:  (rotate90 input) and (flip_h input)
    Result: (□ input) with □ being the hole

    Given:  (replace_color input 3 7) and (replace_color input 5 2)
    Result: (replace_color input □₁ □₂)
    """

    def __init__(self):
        self._hole_counter = 0

    def anti_unify(self, t1: ProgramNode, t2: ProgramNode) -> tuple[ProgramNode, int]:
        """
        Anti-unify two program trees.
        Returns (generalization, number_of_holes).
        """
        self._hole_counter = 0
        result = self._au(t1, t2)
        return result, self._hole_counter

    def _au(self, t1: ProgramNode, t2: ProgramNode) -> ProgramNode:
        # Same primitive
        if isinstance(t1, PrimNode) and isinstance(t2, PrimNode):
            if t1.name == t2.name:
                return PrimNode(t1.name, t1.primitive)

        # Same variable
        if isinstance(t1, VarNode) and isinstance(t2, VarNode):
            if t1.name == t2.name:
                return VarNode(t1.name)

        # Same literal
        if isinstance(t1, LitNode) and isinstance(t2, LitNode):
            if t1.value == t2.value:
                return LitNode(t1.value)

        # Both applications — recurse
        if isinstance(t1, AppNode) and isinstance(t2, AppNode):
            func = self._au(t1.func, t2.func)
            arg = self._au(t1.arg, t2.arg)
            return AppNode(func, arg)

        # Both lambdas with same var — recurse on body
        if isinstance(t1, LamNode) and isinstance(t2, LamNode):
            if t1.var_name == t2.var_name:
                body = self._au(t1.body, t2.body)
                return LamNode(t1.var_name, body)

        # Different structures — introduce a hole
        self._hole_counter += 1
        return VarNode(f"_hole_{self._hole_counter}")


# ──────────────────────────────────────────────────────────────────────
# Corpus rewriting helpers
# ──────────────────────────────────────────────────────────────────────

def _trees_equal(t1: ProgramNode, t2: ProgramNode) -> bool:
    """Structural equality of two ProgramNodes (ignoring primitive object identity)."""
    if type(t1) != type(t2):
        return False
    if isinstance(t1, PrimNode):
        return t1.name == t2.name
    if isinstance(t1, VarNode):
        return t1.name == t2.name
    if isinstance(t1, LitNode):
        return t1.value == t2.value
    if isinstance(t1, AppNode):
        return _trees_equal(t1.func, t2.func) and _trees_equal(t1.arg, t2.arg)
    if isinstance(t1, LamNode):
        return t1.var_name == t2.var_name and _trees_equal(t1.body, t2.body)
    return False


def _contains_holes(node: ProgramNode) -> bool:
    """Return True if any VarNode has a name starting with '_hole_'."""
    if isinstance(node, VarNode):
        return node.name.startswith("_hole_")
    if isinstance(node, AppNode):
        return _contains_holes(node.func) or _contains_holes(node.arg)
    if isinstance(node, LamNode):
        return _contains_holes(node.body)
    return False


def _rewrite_node(
    node: ProgramNode,
    abs_body: ProgramNode,
    abs_prim_node: PrimNode,
) -> ProgramNode:
    """
    Recursively try to replace subtrees of `node` that exactly match
    `abs_body` (no holes) with `abs_prim_node`.

    We only handle zero-hole (exact) matches — this is a correct
    simplification for the current program depth.
    """
    # Try exact match at this node
    if not _contains_holes(abs_body) and _trees_equal(node, abs_body):
        return abs_prim_node

    # Otherwise recurse
    if isinstance(node, AppNode):
        new_func = _rewrite_node(node.func, abs_body, abs_prim_node)
        new_arg = _rewrite_node(node.arg, abs_body, abs_prim_node)
        if new_func is not node.func or new_arg is not node.arg:
            return AppNode(new_func, new_arg)
        return node

    if isinstance(node, LamNode):
        new_body = _rewrite_node(node.body, abs_body, abs_prim_node)
        if new_body is not node.body:
            return LamNode(node.var_name, new_body)
        return node

    return node


def _collect_holes(node: ProgramNode) -> list[str]:
    """Return ordered list of hole variable names in the AST."""
    holes = []
    seen = set()

    def _walk(n):
        if isinstance(n, VarNode) and n.name.startswith("_hole_"):
            if n.name not in seen:
                holes.append(n.name)
                seen.add(n.name)
        elif isinstance(n, AppNode):
            _walk(n.func)
            _walk(n.arg)
        elif isinstance(n, LamNode):
            _walk(n.body)

    _walk(node)
    return holes


def _references_input(node: ProgramNode) -> bool:
    """Return True if the AST contains VarNode('input')."""
    if isinstance(node, VarNode):
        return node.name == "input"
    if isinstance(node, AppNode):
        return _references_input(node.func) or _references_input(node.arg)
    if isinstance(node, LamNode):
        return _references_input(node.body)
    return False


def _concretize_result(t: Type, default: Type) -> Type:
    """
    If the final result type of an Arrow chain is a TypeVariable,
    replace it with `default`.

    e.g. Arrow(tgrid, TypeVariable('_hole_1'))  →  Arrow(tgrid, default)
         Arrow('x, Arrow(tgrid, TypeVariable('h')))  →  Arrow('x, Arrow(tgrid, default))
         tgrid  →  tgrid  (no change)
    """
    if isinstance(t, Arrow):
        new_result = _concretize_result(t.result, default)
        if new_result is not t.result:
            return Arrow(t.arg, new_result)
        return t
    if isinstance(t, TypeVariable):
        return default
    return t


def _unwrap_lambda(node: ProgramNode) -> tuple[ProgramNode, str | None]:
    """If node is a LamNode(var='input', inner), return (inner, 'input'). Else (node, None)."""
    if isinstance(node, LamNode) and node.var_name == "input":
        return node.body, "input"
    return node, None


def _make_abstraction_impl(abs_: Abstraction):
    """
    Create a callable implementation for an abstraction.

    Abstractions have the shape (λinput. body[_hole_1, _hole_2, ...]).
    We unwrap the outer lambda and evaluate the inner body directly
    with env = {"input": grid, "_hole_1": v1, ...}.

    Returns a curried function: hole_1 -> hole_2 -> ... -> input_grid -> grid
    """
    body = abs_.body
    holes = _collect_holes(body)

    # Unwrap outer (λinput. ...) if present — evaluate inner body with env directly
    inner_body, input_var = _unwrap_lambda(body)

    def make_curried(hole_names, hole_values=None):
        if hole_values is None:
            hole_values = {}
        if len(hole_values) < len(hole_names):
            hole_name = hole_names[len(hole_values)]
            def accept_hole(val, _hn=hole_name, _hv=dict(hole_values)):
                _hv[_hn] = val
                return make_curried(hole_names, _hv)
            return accept_hole
        else:
            filled = dict(hole_values)
            if input_var:
                # Body was (λinput. ...) — evaluate inner with input in env
                def impl(input_grid):
                    try:
                        env = {input_var: input_grid, **filled}
                        result = inner_body.evaluate(env)
                        if callable(result):
                            result = result(input_grid)
                        return result
                    except Exception:
                        return input_grid
            else:
                # No outer lambda — evaluate body directly
                def impl(input_grid):
                    try:
                        env = {"input": input_grid, **filled}
                        result = body.evaluate(env)
                        if callable(result):
                            result = result(input_grid)
                        return result
                    except Exception:
                        return input_grid
            return impl

    return make_curried(holes)


# ──────────────────────────────────────────────────────────────────────
# StitchCompressor
# ──────────────────────────────────────────────────────────────────────

class StitchCompressor:
    """
    Stitch-style corpus compression with anti-unification.

    Algorithm:
    1. For each pair of programs, compute anti-unification
    2. Score generalizations by (frequency × holes_saved - definition_cost)
    3. Accept top-K that decrease MDL
    4. Handle cross-system detection

    New capabilities:
    - rewrite_corpus(): rewrites programs to use discovered abstractions
    - register_abstractions_as_primitives(): registers abstractions as real Primitives
    """

    def __init__(
        self,
        library: Library,
        min_frequency: int = 2,
        min_size: int = 2,
        max_holes: int = 3,
        max_abstractions: int = 5,
    ):
        self.library = library
        self.min_frequency = min_frequency
        self.min_size = min_size
        self.max_holes = max_holes
        self.max_abstractions = max_abstractions
        self.au = AntiUnifier()

    # ------------------------------------------------------------------
    def compress(self, programs: list[Program]) -> list[Abstraction]:
        if len(programs) < 2:
            return []

        candidates = []

        # Phase 1: Exact subtree matching (fast)
        candidates.extend(self._exact_subtree_pass(programs))

        # Phase 2: Anti-unification pairs (richer)
        candidates.extend(self._anti_unification_pass(programs))

        # Phase 3: Score and select
        candidates.sort(key=lambda x: -x[1])  # Sort by savings

        # Deduplication: skip patterns already discovered in previous cycles.
        # We compare by STRUCTURAL SIGNATURE (sorted primitives + size + hole count)
        # rather than raw to_str(), because after corpus rewriting, patterns
        # reference abs_N_M names and produce different strings for logically
        # identical abstractions.  The signature captures the computational
        # essence: which base primitives are composed, at what depth, with
        # how many holes.
        existing_signatures = set()
        for existing_abs in self.library.abstractions:
            if existing_abs.body is not None:
                existing_signatures.add(self._structural_signature(existing_abs.body))

        seen_signatures = set()
        accepted = []

        for pattern, savings, sources, systems in candidates:
            if len(accepted) >= self.max_abstractions:
                break

            sig = self._structural_signature(pattern)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            # Skip patterns already in the library from previous cycles
            if sig in existing_signatures:
                continue

            if savings <= 0:
                continue

            # Skip trivial abstractions — patterns that capture shared
            # syntactic skeleton rather than meaningful transformations.
            if self._is_trivial_abstraction(pattern):
                continue

            name = f"abs_{self.library.cycle}_{len(accepted)}"
            abs_ = Abstraction(
                name=name,
                type_signature=self._infer_type(pattern),
                body=pattern,
                source_programs=sources,
                systems_composed=systems,
                reuse_count=len(sources),
                mdl_savings=savings,
            )
            accepted.append(abs_)

        return accepted

    # ------------------------------------------------------------------
    def _is_trivial_abstraction(self, body: ProgramNode) -> bool:
        """
        Reject abstractions that are just syntactic noise.

        A trivial abstraction captures the *shared structure of the language*
        rather than a meaningful transformation.  Examples:
          (λinput. ((_hole_1 input) (_hole_2 _hole_3)))  ← 3 holes, 0 prims
          (λinput. (((_hole_1 input) _hole_2) _hole_3))  ← 3 holes, 0 prims

        Rules:
          1. Zero-hole (exact subtrees) are never trivial — they represent
             concrete, reusable patterns.
          2. For hole-bearing patterns: must have more unique primitives
             than holes.  A pattern with 2 holes and 1 prim is just
             "apply X to stuff" — not a meaningful abstraction.
          3. The body (after unwrapping outer λinput) must contain at least
             2 distinct non-variable, non-hole nodes.
        """
        # Unwrap outer lambda if present
        inner = body
        if isinstance(inner, LamNode):
            inner = inner.body

        holes = _collect_holes(body)
        if not holes:
            # Reject single-primitive wrappers: (λinput. (prim input)) = eta-expansion
            # of a base primitive — not a new abstraction.
            if isinstance(inner, AppNode) and isinstance(inner.func, PrimNode):
                if isinstance(inner.arg, VarNode) and inner.arg.name == "input":
                    return True
                if isinstance(inner.arg, AppNode):
                    # Bare subtree like (prim (arg)) — still might be meaningful
                    pass
            # Also reject bare PrimNodes and VarNodes
            if isinstance(inner, (PrimNode, VarNode, LitNode)):
                return True
            return False

        prims = body.primitives_used()

        # Must have strictly more unique primitives than holes
        if len(prims) <= len(holes):
            return True

        # Must have at least 2 distinct primitives to constitute real structure
        if len(prims) < 2:
            return True

        return False

    # ------------------------------------------------------------------
    def rewrite_corpus(
        self,
        programs: list[Program],
        abstractions: list[Abstraction],
    ) -> list[Program]:
        """
        Rewrite programs using newly discovered abstractions.

        For each abstraction whose body has no holes (exact match),
        replace every matching subtree in every program with a PrimNode
        referring to the abstraction.

        Rewritten programs get source = "enumerator_rewritten" to
        distinguish them from original synthesis outputs.

        Returns a new list of Program objects (originals are not mutated).
        """
        if not abstractions or not programs:
            return list(programs)

        # Build a lightweight Primitive-like stub for each abstraction so we
        # can create PrimNodes.  We use a simple namespace object rather than
        # importing Primitive here, because Primitive requires a full
        # implementation callable — that belongs in register_abstractions_as_primitives.
        class _StubPrimitive:
            def __init__(self, name, type_signature):
                self.name = name
                self.type_signature = type_signature
                self.system = None
                self.description = f"Abstraction {name}"
                self.log_probability = -2.0

            def arity(self):
                from src.spelke_dsl.base import Arrow
                t = self.type_signature
                count = 0
                while isinstance(t, Arrow):
                    count += 1
                    t = t.result
                return count

            def implementation(self, *args):
                return None

        rewritten = []
        for prog in programs:
            current_root = prog.root
            was_rewritten = False

            for abs_ in abstractions:
                # Only rewrite with exact-match (hole-free) abstractions
                if _contains_holes(abs_.body):
                    continue

                stub = _StubPrimitive(abs_.name, abs_.type_signature)
                abs_prim_node = PrimNode(abs_.name, stub)

                new_root = _rewrite_node(current_root, abs_.body, abs_prim_node)
                if new_root is not current_root:
                    current_root = new_root
                    was_rewritten = True

            if was_rewritten:
                rewritten.append(
                    Program(
                        root=current_root,
                        inferred_type=prog.inferred_type,
                        log_likelihood=prog.log_likelihood,
                        log_prior=prog.log_prior,
                        task_id=prog.task_id,
                        source="enumerator_rewritten",
                    )
                )
            else:
                rewritten.append(prog)

        return rewritten

    # ------------------------------------------------------------------
    def register_abstractions_as_primitives(
        self,
        abstractions: list[Abstraction],
        registry,
    ) -> None:
        """
        Register each abstraction as a real callable Primitive in the registry.

        The implementation evaluates the abstraction's AST body with
        {"input": input_grid}.  Already-registered abstractions are skipped.

        Type fixups:
          - Bare subtrees (no outer lambda) that reference VarNode("input")
            get Arrow(tgrid, result_type) so the enumerator treats them as
            grid→... functions, matching the implementation signature.
          - Hole-bearing abstractions: each hole gets Arrow(TypeVariable, ...)
            prepended.  If the result type is a TypeVariable, we substitute
            tgrid (all ARC programs produce grids).
        """
        from src.spelke_dsl.base import Primitive, SpelkeSystem, tgrid

        for abs_ in abstractions:
            if abs_.name in registry._primitives:
                continue

            impl = _make_abstraction_impl(abs_)
            holes = _collect_holes(abs_.body)

            type_sig = abs_.type_signature

            # Fix 1: If the body references "input" but _infer_type didn't
            # produce an Arrow (bare subtree like `(rotate90 input)` → `grid`),
            # wrap it as Arrow(tgrid, result) because the implementation
            # always takes an input_grid argument.
            if not isinstance(type_sig, Arrow) and _references_input(abs_.body):
                type_sig = Arrow(tgrid, type_sig)

            # Fix 2: If the final result type is a TypeVariable (e.g., from
            # hole-bearing patterns like `(_hole_1 input)` → `'_hole_1`),
            # substitute tgrid — all ARC programs must produce grids.
            type_sig = _concretize_result(type_sig, tgrid)

            # Fix 3: Prepend one Arrow(TypeVariable) per hole for curried args.
            for _ in holes:
                type_sig = Arrow(TypeVariable("x"), type_sig)

            prim = Primitive(
                name=abs_.name,
                type_signature=type_sig,
                implementation=impl,
                system=SpelkeSystem.GLUE,
                description=abs_.documentation or f"Learned abstraction: {abs_.name}",
                log_probability=-2.0,
            )
            registry._primitives[abs_.name] = prim

    # ------------------------------------------------------------------
    def _resolve_systems(self, direct_systems: set, node: ProgramNode) -> set:
        """
        Compute the full transitive set of Spelke systems touched by a fragment.

        Direct primitives are already in direct_systems.  Additionally, any
        invented abstraction referenced in the fragment carries its own
        systems_composed — we union those in so that a fragment like

            (abs_0_2 input)          # abs_0_2 bridges OBJECTS+NUMBER
            + (count_objects input)  # count_objects ∈ NUMBER

        is correctly labelled OBJECTS+NUMBER rather than just NUMBER.

        This is the fix for the Carey signature detection: without this,
        every fragment that reuses an earlier cross-system abstraction
        reports zero cross-system primitives.
        """
        systems = set(direct_systems)
        # Build name → systems_composed lookup from current library
        abs_systems: dict[str, set[str]] = {
            a.name: a.systems_composed
            for a in self.library.abstractions
        }
        for prim_name in node.primitives_used():
            if prim_name in abs_systems:
                systems |= abs_systems[prim_name]
        return systems

    # ------------------------------------------------------------------
    def _exact_subtree_pass(self, programs):
        """Find exact recurring subtrees."""
        subtree_counts = Counter()
        subtree_map = {}
        subtree_sources = defaultdict(list)
        subtree_systems = defaultdict(set)

        for prog in programs:
            for st in self._all_subtrees(prog.root):
                key = st.to_str()
                subtree_counts[key] += 1
                subtree_map[key] = st
                subtree_sources[key].append(prog.task_id or "")
                for prim in st.primitives_used():
                    if prim in self.library.base_registry:
                        p = self.library.base_registry[prim]
                        subtree_systems[key].add(p.system.name)

        candidates = []
        for key, count in subtree_counts.items():
            if count < self.min_frequency:
                continue
            node = subtree_map[key]
            if node.size() < self.min_size:
                continue

            savings = node.size() * (count - 1) - count
            if savings > 0:
                # Resolve transitive systems: union direct + nested abstraction systems
                systems = self._resolve_systems(subtree_systems[key], node)
                candidates.append((
                    node, savings,
                    subtree_sources[key],
                    systems,
                ))

        return candidates

    # ------------------------------------------------------------------
    def _anti_unification_pass(self, programs):
        """Find generalizations via pairwise anti-unification."""
        candidates = []
        roots = [(p.root, p.task_id or "") for p in programs]

        # Pairwise AU (limited to avoid O(n²) blowup)
        limit = min(len(roots), 50)

        for i in range(limit):
            for j in range(i + 1, limit):
                t1, tid1 = roots[i]
                t2, tid2 = roots[j]

                gen, n_holes = self.au.anti_unify(t1, t2)

                if n_holes > self.max_holes or n_holes == 0:
                    continue

                gen_size = gen.size()
                if gen_size < self.min_size:
                    continue

                # Check how many other programs match this generalization
                match_count = 0
                match_tids = [tid1, tid2]

                for k, (t, tid) in enumerate(roots):
                    if k == i or k == j:
                        continue
                    _, holes = self.au.anti_unify(t, gen)
                    if holes <= n_holes:
                        match_count += 1
                        match_tids.append(tid)

                total_matches = 2 + match_count
                if total_matches < self.min_frequency:
                    continue

                # MDL savings accounting for holes
                savings = gen_size * (total_matches - 1) - total_matches - n_holes

                if savings > 0:
                    # Collect direct systems from base primitives in the pattern
                    direct = set()
                    pattern_prims = gen.primitives_used()
                    for prim in pattern_prims:
                        if prim in self.library.base_registry:
                            p = self.library.base_registry[prim]
                            direct.add(p.system.name)
                    # Collect hole-filling systems: primitives in source programs
                    # that were generalized away (became holes). These represent
                    # the systems that callers will instantiate the holes with.
                    for t, _ in [roots[i], roots[j]]:
                        for prim in t.primitives_used():
                            if prim not in pattern_prims and prim in self.library.base_registry:
                                p = self.library.base_registry[prim]
                                direct.add(p.system.name)
                    # Union with transitive systems from nested abstractions
                    systems = self._resolve_systems(direct, gen)
                    candidates.append((gen, savings, match_tids, systems))

        return candidates

    # ------------------------------------------------------------------
    def _all_subtrees(self, node):
        result = [node]
        if isinstance(node, AppNode):
            result.extend(self._all_subtrees(node.func))
            result.extend(self._all_subtrees(node.arg))
        elif isinstance(node, LamNode):
            result.extend(self._all_subtrees(node.body))
        return result

    # ------------------------------------------------------------------
    def _infer_type(self, node: ProgramNode) -> Type:
        """
        Infer the result type of a program fragment.

        This walks the AST and uses primitive type signatures to determine
        what type the fragment returns.  Critical: must produce concrete
        types (tgrid, tint, etc.) so the enumerator's _has_concrete_result
        filter accepts invented abstractions.

        Rules:
          - PrimNode: return its type_signature
          - AppNode: if func has type (A → B), return B
          - LamNode: return Arrow(inferred_arg_type, body_type)
          - VarNode("input"): return tgrid (the ARC substrate)
          - VarNode("_hole_N"): return TypeVariable (will be filled by caller)
          - LitNode(int): return tint
          - LitNode(bool): return tbool
        """
        from src.spelke_dsl.base import (
            tgrid, tint, tbool, tcolor, Arrow, TypeVariable, TypeConstructor,
        )

        if isinstance(node, PrimNode):
            return node.primitive.type_signature

        if isinstance(node, AppNode):
            ft = self._infer_type(node.func)
            # Walk through curried Arrows: each AppNode peels off one argument
            if isinstance(ft, Arrow):
                result = ft.result
                # AGGRESSIVE CONCRETIZATION: if the result is a TypeVariable,
                # default to tgrid since all ARC programs produce grids.
                # This fixes the 'x → 'unknown regression.
                if isinstance(result, TypeVariable):
                    return tgrid
                return result
            # If ft is not an Arrow, it may be a TypeVariable from an
            # unresolved abstraction reference.  Default to tgrid.
            if isinstance(ft, TypeVariable):
                return tgrid
            return ft

        if isinstance(node, LamNode):
            body_type = self._infer_type(node.body)
            # CONCRETIZE: if body type resolved to TypeVariable, force tgrid
            if isinstance(body_type, TypeVariable):
                body_type = tgrid
            # For the "input" variable in ARC programs, the arg type is tgrid
            if node.var_name == "input":
                return Arrow(tgrid, body_type)
            # For other variables (e.g., holes), arg type is a TypeVariable
            return Arrow(TypeVariable(node.var_name), body_type)

        if isinstance(node, VarNode):
            # The "input" variable in ARC is always a grid
            if node.name == "input":
                return tgrid
            # Hole variables get a TypeVariable — the caller wraps them in Arrow
            return TypeVariable(node.name)

        if isinstance(node, LitNode):
            if isinstance(node.value, bool):
                return tbool
            if isinstance(node.value, int):
                # In ARC, int literals 0-9 are typically colors
                if 0 <= node.value <= 9:
                    return tcolor
                return tint
            return tint

        # Fallback — should never reach here with well-formed ASTs.
        # Default to tgrid rather than TypeVariable("unknown") so the
        # abstraction isn't invisible to the enumerator.
        return tgrid

    # ------------------------------------------------------------------
    def _expand_abstractions(self, node: ProgramNode) -> ProgramNode:
        """
        Expand all abs_N_M PrimNode references to their full body ASTs.
        This normalizes rewritten programs so size comparisons are apples-to-apples.
        """
        if isinstance(node, PrimNode):
            if node.name.startswith("abs_"):
                for a in self.library.abstractions:
                    if a.name == node.name and a.body is not None:
                        return self._expand_abstractions(a.body)
            return node
        if isinstance(node, AppNode):
            new_func = self._expand_abstractions(node.func)
            new_arg = self._expand_abstractions(node.arg)
            if new_func is not node.func or new_arg is not node.arg:
                return AppNode(new_func, new_arg)
            return node
        if isinstance(node, LamNode):
            new_body = self._expand_abstractions(node.body)
            if new_body is not node.body:
                return LamNode(node.var_name, new_body)
            return node
        return node

    def _structural_signature(self, node: ProgramNode) -> str:
        """
        Compute a structural signature for dedup comparison.

        Normalizes by:
          - Expanding all abs_N_M references to base primitives (for prim set)
          - Computing size on the EXPANDED AST (so rewritten programs with
            abs_N_M single nodes match their full-depth equivalents)
          - Sorting the primitive set for order-independence
          - Including hole count

        Two patterns with the same base primitives, same expanded size, and
        same hole count are considered duplicates regardless of whether they
        reference earlier abstractions or write out the full subtree.
        """
        expanded = self._expand_abstractions(node)
        base_prims = self._collect_base_primitives(node)
        holes = _collect_holes(node)
        return f"prims={'|'.join(sorted(base_prims))}_size={expanded.size()}_holes={len(holes)}"

    def _collect_base_primitives(self, node: ProgramNode) -> set:
        """
        Collect all BASE (non-abstraction) primitives in a pattern,
        recursively expanding any abs_N_M references through the library.
        """
        result = set()
        if isinstance(node, PrimNode):
            if node.name.startswith("abs_"):
                # Expand: find this abstraction in the library and recurse
                for a in self.library.abstractions:
                    if a.name == node.name and a.body is not None:
                        result |= self._collect_base_primitives(a.body)
                        break
                else:
                    result.add(node.name)  # Can't expand, keep as-is
            else:
                result.add(node.name)
        elif isinstance(node, AppNode):
            result |= self._collect_base_primitives(node.func)
            result |= self._collect_base_primitives(node.arg)
        elif isinstance(node, LamNode):
            result |= self._collect_base_primitives(node.body)
        return result
