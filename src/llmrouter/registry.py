"""Name -> class registries used to pick routers and embedders from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A small name -> class mapping filled by the ``register`` decorator."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Class decorator: register ``cls`` under ``name`` and set ``cls.registry_name``."""

        def decorator(cls: type[T]) -> type[T]:
            existing = self._items.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"{self.kind} '{name}' is already registered to {existing}")
            self._items[name] = cls
            cls.registry_name = name  # type: ignore[attr-defined]
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown {self.kind} '{name}'; available: {available}") from None

    def create(self, name: str, **kwargs: Any) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


ROUTERS: Registry[Any] = Registry("router")
EMBEDDERS: Registry[Any] = Registry("embedder")
