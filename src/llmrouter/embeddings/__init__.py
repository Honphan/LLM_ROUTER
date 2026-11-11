"""Text embedders and their on-disk cache. Importing this package registers all providers."""

from llmrouter.embeddings import providers as providers
from llmrouter.embeddings.base import Embedder
from llmrouter.embeddings.cache import embed_cached

__all__ = ["Embedder", "embed_cached"]
