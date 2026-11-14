"""Routing algorithms. Importing this package registers every router in ``ROUTERS``."""

from llmrouter.routers.base import Router
from llmrouter.routers.random import router as _random_router  # noqa: F401
from llmrouter.routers.sw_ranking import router as _sw_ranking_router  # noqa: F401

__all__ = ["Router"]
