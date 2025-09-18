"""
language.py — LAPS-Style Language Conditioning Module

Implements language-guided abstraction selection and search prioritization,
inspired by Wong et al. (2021) LAPS and Grand et al. (2024) LILO.

In Carey's framework, language is the primary placeholder-supplying mechanism:
words like "one, two, three" provide the scaffolding that eventually gets
filled with conceptual content via modeling processes. This module bridges
natural-language task descriptions (from LARC annotations or LLM descriptions)
to the DSL primitive space.

Three capabilities:
  1. Task description → primitive prior: given a text description, predict
     which primitives are likely relevant (language-guided search).
  2. Abstraction documentation: auto-document invented abstractions in
     natural language to improve interpretability and enable LLM-based
     search.
  3. Language-conditioned dream generation: generate dreams that match
     linguistic task descriptions, biasing the recognition network.

Usage:
    from src.engine.language import LanguageConditioner
    lc = LanguageConditioner(library)
    priors = lc.description_to_priors("rotate the largest object 90 degrees")
    # priors = {"rotate90": 0.8, "obj_largest": 0.9, "extract_objects": 0.85, ...}
"""

from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import Optional
from src.spelke_dsl.base import PrimitiveRegistry, SpelkeSystem


# ──────────────────────────────────────────────────────────────────────
# Keyword → Primitive Mapping (hand-coded; could be replaced by LLM)
# ──────────────────────────────────────────────────────────────────────

# Maps natural language keywords to primitive names + relevance weight
KEYWORD_PRIM_MAP: dict[str, list[tuple[str, float]]] = {
    # OBJECTS keywords
    "object": [("extract_objects", 0.9), ("render_objects", 0.7), ("obj_largest", 0.5)],
    "objects": [("extract_objects", 0.9), ("render_objects", 0.7), ("count_objects", 0.6)],
    "largest": [("obj_largest", 0.95), ("extract_objects", 0.8)],
    "smallest": [("obj_smallest", 0.95), ("extract_objects", 0.8)],
    "biggest": [("obj_largest", 0.9), ("extract_objects", 0.8)],
    "extract": [("extract_objects", 0.9)],
    "isolate": [("extract_objects", 0.8), ("render_object", 0.7)],
    "remove": [("render_objects_on", 0.7), ("extract_objects", 0.6)],
    "filter": [("obj_filter_color", 0.8), ("extract_objects", 0.7)],
    "contact": [("obj_touching", 0.9)],
    "touching": [("obj_touching", 0.9)],
    "gravity": [("gravity_down", 0.8), ("gravity_up", 0.5)],
    "fall": [("gravity_down", 0.9)],
    "slide": [("gravity_down", 0.7), ("gravity_left", 0.5), ("gravity_right", 0.5)],
    "fill": [("fill_interior", 0.9), ("flood_fill", 0.7)],
    "flood": [("flood_fill", 0.9)],
    "interior": [("fill_interior", 0.9)],

    # FORMS keywords
    "rotate": [("rotate90", 0.8), ("rotate180", 0.6), ("rotate270", 0.5)],
    "rotation": [("rotate90", 0.8), ("rotate180", 0.6)],
    "flip": [("flip_h", 0.7), ("flip_v", 0.7)],
    "mirror": [("flip_h", 0.8), ("flip_v", 0.8), ("sym_h_complete", 0.5)],
    "reflect": [("flip_h", 0.7), ("flip_v", 0.7)],
    "horizontal": [("flip_h", 0.7), ("hstack_grid", 0.4)],
    "vertical": [("flip_v", 0.7), ("vstack_grid", 0.4)],
    "transpose": [("transpose", 0.95)],
    "diagonal": [("transpose", 0.6)],
    "symmetry": [("sym_h_complete", 0.8), ("sym_v_complete", 0.8)],
    "symmetric": [("sym_h_complete", 0.7), ("sym_v_complete", 0.7)],
    "tile": [("tile", 0.9), ("extract_tile", 0.6)],
    "repeat": [("tile", 0.8)],
    "pattern": [("extract_tile", 0.7), ("tile", 0.5)],
    "scale": [("scale_up", 0.8), ("scale_down", 0.6)],
    "enlarge": [("scale_up", 0.9)],
    "shrink": [("scale_down", 0.9)],
    "resize": [("scale_up", 0.7), ("scale_down", 0.7)],
    "border": [("make_border", 0.9)],
    "frame": [("make_border", 0.8)],
    "stack": [("vstack_grid", 0.7), ("hstack_grid", 0.7)],
    "concatenate": [("vstack_grid", 0.6), ("hstack_grid", 0.6)],
    "overlay": [("overlay", 0.9)],
    "combine": [("overlay", 0.7), ("vstack_grid", 0.4)],
    "crop": [("crop", 0.9), ("extract_tile", 0.5)],

    # NUMBER keywords
    "count": [("count_objects", 0.9), ("extract_objects", 0.6)],
    "number": [("count_objects", 0.7)],
    "size": [("obj_size", 0.8), ("obj_largest", 0.5)],

    # COLOR keywords
    "color": [("replace_color", 0.7)],
    "recolor": [("replace_color", 0.9)],
    "replace": [("replace_color", 0.8)],

    # AGENTS keywords
    "move": [("move_to_goal", 0.7), ("move_toward", 0.6)],
    "path": [("trace_path", 0.9)],
    "goal": [("move_to_goal", 0.8), ("infer_goal_dir", 0.6)],
    "direction": [("infer_goal_dir", 0.7)],
    "transport": [("transport_object", 0.9)],
    "carry": [("transport_object", 0.8)],

    # PLACES keywords
    "quadrant": [("extract_quadrant", 0.9)],
    "quarter": [("extract_quadrant", 0.8)],
    "center": [("get_center", 0.8), ("distance_from_center", 0.5)],
    "edge": [("get_border", 0.7), ("distance_from_edge", 0.6)],
    "distance": [("distance_from_color", 0.7), ("distance_from_edge", 0.5)],
    "region": [("segment_by_axes", 0.7), ("extract_quadrant", 0.5)],
    "row": [("get_row", 0.8)],
    "column": [("get_col", 0.8)],
}


class LanguageConditioner:
    """
    Language-guided primitive prior computation.

    Given a natural-language task description, produces a probability
    distribution over primitives to guide search.
    """

    def __init__(self, registry: PrimitiveRegistry):
        self.registry = registry
        self._all_prim_names = set(registry.names())
        self._llm_cache: dict[str, dict[str, float]] = {}

    def description_to_priors(
        self, description: str, base_prior: float = 0.1
    ) -> dict[str, float]:
        """
        Convert a natural-language description to primitive relevance scores.

        Returns a dict mapping primitive names to [0, 1] scores.
        Primitives not mentioned get `base_prior`.
        """
        description_lower = description.lower()
        words = set(re.findall(r'\b\w+\b', description_lower))

        priors = {name: base_prior for name in self._all_prim_names}

        for word in words:
            if word in KEYWORD_PRIM_MAP:
                for prim_name, weight in KEYWORD_PRIM_MAP[word]:
                    if prim_name in priors:
                        # Max over multiple keyword hits
                        priors[prim_name] = max(priors[prim_name], weight)

        return priors

    def llm_description_to_priors(
        self, description: str, model: str = "claude-sonnet-4-6"
    ) -> dict[str, float]:
        """
        Use an LLM to map a description to primitive relevance scores.
        Requires ANTHROPIC_API_KEY environment variable.

        Caches results to avoid redundant API calls.
        """
        cache_key = description.strip().lower()
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        try:
            import anthropic
        except ImportError:
            return self.description_to_priors(description)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return self.description_to_priors(description)

        prim_list = ", ".join(sorted(self._all_prim_names)[:50])

        prompt = f"""Given this ARC task description, rate the relevance (0.0-1.0) of each DSL primitive.

Task: {description}

Available primitives (subset): {prim_list}

Return JSON mapping primitive names to relevance scores (0.0-1.0). Only include primitives with score > 0.2.
Example: {{"rotate90": 0.9, "extract_objects": 0.8}}"""

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            llm_priors = json.loads(raw)
            # Merge with base priors
            base = {name: 0.1 for name in self._all_prim_names}
            for k, v in llm_priors.items():
                if k in base:
                    base[k] = max(base[k], float(v))

            self._llm_cache[cache_key] = base
            return base

        except Exception:
            return self.description_to_priors(description)

    def document_abstraction(
        self, name: str, body_str: str, systems: list[str],
        reuse_count: int, model: str = "claude-sonnet-4-6"
    ) -> str:
        """
        Auto-document an invented abstraction using an LLM.
        Returns a 1-sentence natural-language description.
        """
        try:
            import anthropic
        except ImportError:
            return f"Abstraction {name} combining {'+'.join(systems)}"

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return f"Abstraction {name} combining {'+'.join(systems)}"

        prompt = f"""Name a learned program abstraction in one clear sentence.

Abstraction: {name}
Body: {body_str}
Systems used: {', '.join(systems)}
Reused in {reuse_count} programs

Give ONLY the 1-sentence name/description, no explanation."""

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            return f"Abstraction {name} combining {'+'.join(systems)}"


# ──────────────────────────────────────────────────────────────────────
# LARC Annotation Loader
# ──────────────────────────────────────────────────────────────────────

def load_larc_descriptions(
    larc_dir: str = "data/larc",
) -> dict[str, str]:
    """
    Load LARC natural-language task descriptions (Acquaviva et al. 2022).
    Returns dict mapping task_id → description string.

    Expected format: larc_dir/task_id.json with key "description" or "text".
    """
    descriptions = {}
    larc_path = Path(larc_dir)
    if not larc_path.exists():
        return descriptions

    for json_file in larc_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            task_id = json_file.stem
            desc = data.get("description", data.get("text", ""))
            if isinstance(desc, list):
                desc = " ".join(desc)
            if desc:
                descriptions[task_id] = desc
        except Exception:
            continue

    return descriptions
