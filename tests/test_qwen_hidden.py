from __future__ import annotations

import pytest

from graph_model.mlx_native.qwen_hidden import _resolve_backbone, parse_layer_specs
from graph_model.provider import ProviderError


def test_layer_selectors_support_final_integer_negative_and_percent() -> None:
    assert parse_layer_specs(("final", "0", "-1", "50%"), 8) == (
        ("final", None),
        ("layer:0", 0),
        ("layer:7", 7),
        ("layer:3", 3),
    )


def test_layer_selector_deduplicates_resolved_indices() -> None:
    assert parse_layer_specs(("50%", "3", "final", "final"), 8) == (
        ("layer:3", 3),
        ("final", None),
    )


def test_layer_selector_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="outside"):
        parse_layer_specs(("9",), 8)
    with pytest.raises(ValueError, match="percentage"):
        parse_layer_specs(("0%",), 8)


def test_backbone_resolution_prefers_nested_language_model() -> None:
    class Backbone:
        embed_tokens = object()
        norm = object()
        layers = [object()]

        def __call__(self, *args, **kwargs):
            del args, kwargs

    class TextModel:
        model = Backbone()

    class Outer:
        language_model = TextModel()

    backbone, source = _resolve_backbone(Outer())
    assert isinstance(backbone, Backbone)
    assert source == "language_model.model"


def test_backbone_resolution_rejects_logits_only_models() -> None:
    with pytest.raises(ProviderError, match="does not expose"):
        _resolve_backbone(object())
