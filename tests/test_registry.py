from __future__ import annotations

import pytest

from llmrouter.registry import Registry


def test_register_get_create() -> None:
    registry: Registry[object] = Registry("thing")

    @registry.register("box")
    class Box:
        def __init__(self, size: int = 1) -> None:
            self.size = size

    assert registry.get("box") is Box
    assert Box.registry_name == "box"  # type: ignore[attr-defined]
    assert registry.create("box", size=3).size == 3  # type: ignore[attr-defined]
    assert registry.names() == ["box"]
    assert "box" in registry


def test_unknown_name_lists_available() -> None:
    registry: Registry[object] = Registry("thing")
    registry.register("a")(type("A", (), {}))
    with pytest.raises(KeyError, match="available: a"):
        registry.get("missing")


def test_duplicate_name_rejected() -> None:
    registry: Registry[object] = Registry("thing")
    registry.register("a")(type("A", (), {}))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a")(type("B", (), {}))
