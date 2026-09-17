"""Semantic analysis layer (Phase 3).

Ranks a word's WordNet senses by how similar their glosses are to the
surrounding context, producing *evidence* for the later LLM layer rather than
a decision about what a word means.
"""

from ambisense.semantic.context_resolver import SemanticAnalyzer
from ambisense.semantic.embeddings import (
    EmbeddingBackend,
    SpacyEmbeddingBackend,
    cosine_similarity,
)
from ambisense.semantic.wordnet_senses import (
    count_available_senses,
    gloss_text,
    retrieve_senses,
)

__all__ = [
    "SemanticAnalyzer",
    "EmbeddingBackend",
    "SpacyEmbeddingBackend",
    "cosine_similarity",
    "retrieve_senses",
    "count_available_senses",
    "gloss_text",
]
