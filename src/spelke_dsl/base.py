"""
base.py — Type system, Primitive representation, and Registry.

This module defines the typed lambda-calculus substrate that all Spelke
primitives live in. Compatible with DreamCoder/LILO's type inference system.

Design Note (DESIGN_NOTES.md §1):
    We use a simply-typed lambda calculus with parametric polymorphism.
    Each primitive is a typed function with metadata indicating which
    Spelke core system it belongs to. This enables:
    (a) Tracking cross-system composition (the "Carey signature")
    (b) Matched-cardinality baseline construction
    (c) Principled type-checking during program synthesis
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


# ──────────────────────────────────────────────────────────────────────
# Spelke Core Knowledge System Labels
# ──────────────────────────────────────────────────────────────────────

class SpelkeSystem(Enum):
    """The six core knowledge systems (Spelke 2022, 2023 BBS précis)."""
    OBJECTS = auto()    # Cohesion, continuity, contact
    AGENTS = auto()     # Goal-directedness, equifinality, rational efficiency
    PERSONS = auto()    # Face preference, native-language recognition, like-others
    PLACES = auto()     # Geocentric layout, geometric reorientation
    FORMS = auto()      # Scale-invariant shape geometry
    NUMBER = auto()     # ANS + parallel individuation (OTS)
    GLUE = auto()       # Generic combinators (not a Spelke system)
    ANALOGY = auto()    # Cross-domain transfer (Gentner)
    NSM = auto()        # Wierzbicka logical/causal operators


# ──────────────────────────────────────────────────────────────────────
# Type System
# ──────────────────────────────────────────────────────────────────────

class Type:
    """Base class for all types in the DSL."""

    def __eq__(self, other):
        return type(self) == type(other) and self._key() == other._key()

    def __hash__(self):
        return hash((type(self).__name__, self._key()))

    def _key(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def returns(self) -> Type:
        """Get the return type (self for base types, result for arrows)."""
        return self

    def arguments(self) -> list[Type]:
        """Get argument types (empty for base types)."""
        return []

    def is_polymorphic(self) -> bool:
        return False


class TypeVariable(Type):
    """A polymorphic type variable (e.g., 'a, 'b)."""

    def __init__(self, name: str):
        self.name = name

    def _key(self):
        return self.name

    def __repr__(self):
        return f"'{self.name}"

    def is_polymorphic(self) -> bool:
        return True


class TypeConstructor(Type):
    """A concrete type like int, bool, grid, object, etc."""

    def __init__(self, name: str):
        self.name = name

    def _key(self):
        return self.name

    def __repr__(self):
        return self.name


class Arrow(Type):
    """Function type: argument -> result."""

    def __init__(self, arg: Type, result: Type):
        self.arg = arg
        self.result = result

    def _key(self):
        return (self.arg, self.result)

    def __repr__(self):
        arg_str = f"({self.arg})" if isinstance(self.arg, Arrow) else str(self.arg)
        return f"{arg_str} → {self.result}"

    def returns(self) -> Type:
        if isinstance(self.result, Arrow):
            return self.result.returns()
        return self.result

    def arguments(self) -> list[Type]:
        args = [self.arg]
        if isinstance(self.result, Arrow):
            args.extend(self.result.arguments())
        return args

    def is_polymorphic(self) -> bool:
        return self.arg.is_polymorphic() or self.result.is_polymorphic()


class ListType(Type):
    """Parameterized list type: list[T]."""

    def __init__(self, element_type: Type):
        self.element_type = element_type

    def _key(self):
        return self.element_type

    def __repr__(self):
        return f"list[{self.element_type}]"

    def is_polymorphic(self) -> bool:
        return self.element_type.is_polymorphic()


class PairType(Type):
    """Parameterized pair type: (A, B)."""

    def __init__(self, first: Type, second: Type):
        self.first = first
        self.second = second

    def _key(self):
        return (self.first, self.second)

    def __repr__(self):
        return f"({self.first}, {self.second})"

    def is_polymorphic(self) -> bool:
        return self.first.is_polymorphic() or self.second.is_polymorphic()


class OptionType(Type):
    """Optional type: maybe[T]."""

    def __init__(self, inner: Type):
        self.inner = inner

    def _key(self):
        return self.inner

    def __repr__(self):
        return f"maybe[{self.inner}]"

    def is_polymorphic(self) -> bool:
        return self.inner.is_polymorphic()


# ──────────────────────────────────────────────────────────────────────
# Standard Types — the ARC-AGI domain types
# ──────────────────────────────────────────────────────────────────────

tint = TypeConstructor("int")
tbool = TypeConstructor("bool")
tcolor = TypeConstructor("color")        # 0-9 color palette in ARC
tpoint = TypeConstructor("point")        # (row, col) coordinate
tdir = TypeConstructor("dir")            # Direction: up/down/left/right/diag
tgrid = TypeConstructor("grid")          # 2D color grid (the ARC substrate)
tobject = TypeConstructor("object")      # Extracted object (connected component)
tshape = TypeConstructor("shape")        # Normalized shape (translation-invariant)
taxis = TypeConstructor("axis")          # Symmetry axis
trect = TypeConstructor("rect")          # Bounding rectangle
tline = TypeConstructor("line")          # Line segment
tmapping = TypeConstructor("mapping")    # Structural alignment mapping
tprogram = TypeConstructor("program")    # Program reference (for higher-order)
tunit = TypeConstructor("unit")          # Unit / void type
tfloat = TypeConstructor("float")        # Float for ANS

# Polymorphic type variables
t0 = TypeVariable("a")
t1 = TypeVariable("b")
t2 = TypeVariable("c")

# Convenience constructors
def tlist(t: Type) -> ListType:
    return ListType(t)

def tpair(a: Type, b: Type) -> PairType:
    return PairType(a, b)

def tmaybe(t: Type) -> OptionType:
    return OptionType(t)


# ──────────────────────────────────────────────────────────────────────
# Type Unification (for type inference during synthesis)
# ──────────────────────────────────────────────────────────────────────

class UnificationError(Exception):
    pass


def unify(t1: Type, t2: Type, substitution: dict[str, Type] | None = None) -> dict[str, Type]:
    """
    Unify two types, returning a substitution mapping type variables to concrete types.
    Raises UnificationError if types are incompatible.
    """
    if substitution is None:
        substitution = {}

    t1 = _apply_substitution(t1, substitution)
    t2 = _apply_substitution(t2, substitution)

    if isinstance(t1, TypeVariable):
        if t1 == t2:
            return substitution
        if _occurs_in(t1.name, t2):
            raise UnificationError(f"Occurs check failed: {t1} in {t2}")
        substitution[t1.name] = t2
        return substitution

    if isinstance(t2, TypeVariable):
        return unify(t2, t1, substitution)

    if isinstance(t1, TypeConstructor) and isinstance(t2, TypeConstructor):
        if t1.name == t2.name:
            return substitution
        raise UnificationError(f"Cannot unify {t1} with {t2}")

    if isinstance(t1, Arrow) and isinstance(t2, Arrow):
        substitution = unify(t1.arg, t2.arg, substitution)
        return unify(t1.result, t2.result, substitution)

    if isinstance(t1, ListType) and isinstance(t2, ListType):
        return unify(t1.element_type, t2.element_type, substitution)

    if isinstance(t1, PairType) and isinstance(t2, PairType):
        substitution = unify(t1.first, t2.first, substitution)
        return unify(t1.second, t2.second, substitution)

    if isinstance(t1, OptionType) and isinstance(t2, OptionType):
        return unify(t1.inner, t2.inner, substitution)

    raise UnificationError(f"Cannot unify {t1} with {t2}")


def _apply_substitution(t: Type, sub: dict[str, Type]) -> Type:
    if isinstance(t, TypeVariable):
        if t.name in sub:
            return _apply_substitution(sub[t.name], sub)
        return t
    if isinstance(t, Arrow):
        return Arrow(_apply_substitution(t.arg, sub), _apply_substitution(t.result, sub))
    if isinstance(t, ListType):
        return ListType(_apply_substitution(t.element_type, sub))
    if isinstance(t, PairType):
        return PairType(_apply_substitution(t.first, sub), _apply_substitution(t.second, sub))
    if isinstance(t, OptionType):
        return OptionType(_apply_substitution(t.inner, sub))
    return t


def _occurs_in(var_name: str, t: Type) -> bool:
    if isinstance(t, TypeVariable):
        return t.name == var_name
    if isinstance(t, Arrow):
        return _occurs_in(var_name, t.arg) or _occurs_in(var_name, t.result)
    if isinstance(t, ListType):
        return _occurs_in(var_name, t.element_type)
    if isinstance(t, PairType):
        return _occurs_in(var_name, t.first) or _occurs_in(var_name, t.second)
    if isinstance(t, OptionType):
        return _occurs_in(var_name, t.inner)
    return False


def fresh_type_variables(t: Type, prefix: str = "_t") -> tuple[Type, dict[str, Type]]:
    """Replace all type variables with fresh ones to avoid capture."""
    mapping = {}
    counter = [0]

    def freshen(typ: Type) -> Type:
        if isinstance(typ, TypeVariable):
            if typ.name not in mapping:
                mapping[typ.name] = TypeVariable(f"{prefix}{counter[0]}")
                counter[0] += 1
            return mapping[typ.name]
        if isinstance(typ, Arrow):
            return Arrow(freshen(typ.arg), freshen(typ.result))
        if isinstance(typ, ListType):
            return ListType(freshen(typ.element_type))
        if isinstance(typ, PairType):
            return PairType(freshen(typ.first), freshen(typ.second))
        if isinstance(typ, OptionType):
            return OptionType(freshen(typ.inner))
        return typ

    return freshen(t), mapping


# ──────────────────────────────────────────────────────────────────────
# Primitive Definition
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Primitive:
    """
    A single typed primitive in the Spelke DSL.

    Each primitive carries:
    - name: unique identifier (used in programs)
    - type_signature: the simply-typed λ-calculus type
    - implementation: the actual Python callable
    - system: which Spelke core system this belongs to
    - description: human-readable documentation
    - spelke_properties: which of the six Spelke properties it exhibits
    """
    name: str
    type_signature: Type
    implementation: Callable
    system: SpelkeSystem
    description: str = ""
    spelke_properties: list[str] = field(default_factory=list)
    log_probability: float = 0.0  # Prior weight in PCFG

    def __call__(self, *args, **kwargs):
        return self.implementation(*args, **kwargs)

    def __repr__(self):
        return f"Primitive({self.name}: {self.type_signature}, system={self.system.name})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Primitive):
            return self.name == other.name
        return False

    def arity(self) -> int:
        return len(self.type_signature.arguments())

    def fingerprint(self) -> str:
        """Content hash for versioning."""
        content = f"{self.name}:{self.type_signature}:{self.system.name}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": str(self.type_signature),
            "system": self.system.name,
            "description": self.description,
            "arity": self.arity(),
            "log_probability": self.log_probability,
        }


# ──────────────────────────────────────────────────────────────────────
# Primitive Registry
# ──────────────────────────────────────────────────────────────────────

class PrimitiveRegistry:
    """
    Registry of all typed primitives available to the synthesis engine.

    Manages:
    - Registration and lookup of primitives
    - Cardinality tracking per Spelke system (for matched baselines)
    - Serialization for reproducibility auditing
    """

    def __init__(self):
        self._primitives: dict[str, Primitive] = {}
        self._by_system: dict[SpelkeSystem, list[Primitive]] = {s: [] for s in SpelkeSystem}

    def register(self, primitive: Primitive) -> None:
        """Register a primitive. Raises ValueError on duplicate names."""
        if primitive.name in self._primitives:
            raise ValueError(f"Duplicate primitive name: {primitive.name}")
        self._primitives[primitive.name] = primitive
        self._by_system[primitive.system].append(primitive)

    def get(self, name: str) -> Primitive:
        """Retrieve a primitive by name."""
        if name not in self._primitives:
            raise KeyError(f"Unknown primitive: {name}")
        return self._primitives[name]

    def __getitem__(self, name: str) -> Primitive:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._primitives

    def __len__(self) -> int:
        return len(self._primitives)

    def __iter__(self):
        return iter(self._primitives.values())

    def all_primitives(self) -> list[Primitive]:
        return list(self._primitives.values())

    def by_system(self, system: SpelkeSystem) -> list[Primitive]:
        return list(self._by_system[system])

    def system_cardinality(self) -> dict[str, int]:
        """Cardinality per system — critical for matched-baseline construction."""
        return {s.name: len(ps) for s, ps in self._by_system.items() if ps}

    def total_cardinality(self) -> int:
        return len(self._primitives)

    def names(self) -> list[str]:
        return list(self._primitives.keys())

    def type_signatures(self) -> dict[str, Type]:
        return {name: p.type_signature for name, p in self._primitives.items()}

    def filter_by_return_type(self, return_type: Type) -> list[Primitive]:
        """Find all primitives that return a given type."""
        results = []
        for p in self._primitives.values():
            if p.type_signature.returns() == return_type:
                results.append(p)
        return results

    def to_pcfg_weights(self) -> dict[str, float]:
        """Export log-probabilities for PCFG prior."""
        return {name: p.log_probability for name, p in self._primitives.items()}

    def manifest(self) -> dict:
        """
        Full manifest for reproducibility — DESIGN_NOTES.md §2.
        Every design choice must be auditable.
        """
        return {
            "total_primitives": self.total_cardinality(),
            "cardinality_by_system": self.system_cardinality(),
            "primitives": [p.to_dict() for p in self._primitives.values()],
            "fingerprint": self._registry_fingerprint(),
        }

    def _registry_fingerprint(self) -> str:
        content = json.dumps(
            sorted([p.fingerprint() for p in self._primitives.values()])
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_json(self, path: str) -> None:
        """Serialize the full registry manifest."""
        import json
        with open(path, "w") as f:
            json.dump(self.manifest(), f, indent=2)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"PrimitiveRegistry: {self.total_cardinality()} primitives"]
        for system, prims in self._by_system.items():
            if prims:
                lines.append(f"  {system.name}: {len(prims)} primitives")
                for p in prims:
                    lines.append(f"    {p.name}: {p.type_signature}")
        return "\n".join(lines)
