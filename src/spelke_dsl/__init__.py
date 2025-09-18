"""
Spelke DSL — Core Knowledge Primitive Libraries

The first computational instantiation of Spelke's six core knowledge systems
as a typed domain-specific language for program induction.

Phase 1 (Active): OBJECTS, FORMS, NUMBER
Phase 2 (New):    AGENTS, PLACES, PERSONS
Glue:             Generic combinators + Gentner analogy
"""

from src.spelke_dsl.base import (
    Primitive,
    PrimitiveRegistry,
    SpelkeSystem,
    Type,
    TypeConstructor,
    Arrow,
    tgrid,
    tobject,
    tcolor,
    tint,
    tbool,
    tpoint,
    tdir,
    tshape,
    tlist,
    tpair,
)
from src.spelke_dsl.l_objects import register_object_primitives
from src.spelke_dsl.l_forms import register_form_primitives
from src.spelke_dsl.l_number import register_number_primitives
from src.spelke_dsl.l_agents import register_agent_primitives
from src.spelke_dsl.l_places import register_place_primitives
from src.spelke_dsl.l_persons import register_persons_primitives
from src.spelke_dsl.glue import register_glue_primitives
from src.spelke_dsl.analogy import register_analogy_primitives


def build_spelke_library(
    include_agents: bool = False,
    include_places: bool = False,
    include_persons: bool = False,
) -> PrimitiveRegistry:
    """
    Build the complete Spelke-initialized primitive library.

    Args:
        include_agents: Include AGENTS module (Phase 2)
        include_places: Include PLACES module (Phase 2)
        include_persons: Include PERSONS module (Phase 2)

    Returns a PrimitiveRegistry containing all active sub-libraries.
    """
    registry = PrimitiveRegistry()

    # Phase 1: Three core modules
    register_object_primitives(registry)
    register_form_primitives(registry)
    register_number_primitives(registry)

    # Phase 2: Extended modules (opt-in for backward compatibility)
    if include_agents:
        register_agent_primitives(registry)
    if include_places:
        register_place_primitives(registry)
    if include_persons:
        register_persons_primitives(registry)

    # Generic combinators + NSM operators
    register_glue_primitives(registry)

    # Cross-domain analogy primitive
    register_analogy_primitives(registry)

    return registry


def build_full_spelke_library() -> PrimitiveRegistry:
    """Build with all six systems active (6 implemented + GLUE)."""
    return build_spelke_library(include_agents=True, include_places=True, include_persons=True)


__all__ = [
    "build_spelke_library",
    "build_full_spelke_library",
    "Primitive",
    "PrimitiveRegistry",
    "SpelkeSystem",
    "Type",
    "TypeConstructor",
    "Arrow",
    "tgrid",
    "tobject",
    "tcolor",
    "tint",
    "tbool",
    "tpoint",
    "tdir",
    "tshape",
    "tlist",
    "tpair",
]
