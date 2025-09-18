"""Tests for the language conditioning module."""
import pytest
from src.spelke_dsl.base import PrimitiveRegistry
from src.spelke_dsl import build_spelke_library
from src.engine.language import LanguageConditioner, KEYWORD_PRIM_MAP


@pytest.fixture
def conditioner():
    reg = build_spelke_library()
    return LanguageConditioner(reg)


class TestKeywordMapping:
    def test_rotate_description(self, conditioner):
        priors = conditioner.description_to_priors("rotate the grid 90 degrees")
        assert priors.get("rotate90", 0) > 0.5
        assert priors.get("rotate180", 0) > 0.3

    def test_object_description(self, conditioner):
        priors = conditioner.description_to_priors("extract the largest object")
        assert priors.get("extract_objects", 0) > 0.7
        assert priors.get("obj_largest", 0) > 0.7

    def test_combined_description(self, conditioner):
        priors = conditioner.description_to_priors(
            "rotate the largest object 90 degrees"
        )
        assert priors.get("rotate90", 0) > 0.5
        assert priors.get("obj_largest", 0) > 0.7
        assert priors.get("extract_objects", 0) > 0.5

    def test_unknown_words(self, conditioner):
        priors = conditioner.description_to_priors("xyzzy frobulate the qux")
        # All priors should be at base level
        for name, score in priors.items():
            assert score == pytest.approx(0.1, abs=0.01)

    def test_color_keywords(self, conditioner):
        priors = conditioner.description_to_priors("recolor the cells")
        assert priors.get("replace_color", 0) > 0.5

    def test_gravity_keywords(self, conditioner):
        priors = conditioner.description_to_priors("let objects fall down")
        assert priors.get("gravity_down", 0) > 0.5

    def test_scale_keywords(self, conditioner):
        priors = conditioner.description_to_priors("enlarge the pattern by 2x")
        assert priors.get("scale_up", 0) > 0.5


class TestKeywordCoverage:
    def test_all_keywords_have_valid_prims(self):
        """Every keyword should map to at least one primitive name."""
        for keyword, prim_list in KEYWORD_PRIM_MAP.items():
            assert len(prim_list) > 0, f"Keyword '{keyword}' has no primitives"
            for name, weight in prim_list:
                assert 0 <= weight <= 1, f"Weight for {name} out of range: {weight}"
                assert isinstance(name, str), f"Primitive name not str: {name}"

    def test_keyword_map_not_empty(self):
        assert len(KEYWORD_PRIM_MAP) > 30
