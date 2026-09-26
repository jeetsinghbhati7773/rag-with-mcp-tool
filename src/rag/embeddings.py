from __future__ import annotations

import logging
import re
from typing import List
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Cache global model instance
_EMBED_MODEL = None


def get_embedding_model():
    """
    Returns cached SentenceTransformer model instance ('all-MiniLM-L6-v2').
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded SentenceTransformer model 'all-MiniLM-L6-v2' successfully.")
        except Exception as e:
            logger.warning("Could not load SentenceTransformer: %s", e)
            _EMBED_MODEL = None
    return _EMBED_MODEL


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible embedding wrapper around SentenceTransformers 'all-MiniLM-L6-v2'.
    Fallback to deterministic lexical vector embedding if SentenceTransformer package is unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = get_embedding_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, convert_to_numpy=True)
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error("SentenceTransformer encoding failed: %s", e)

        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        model = get_embedding_model()
        if model is not None:
            try:
                embedding = model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.error("SentenceTransformer query encoding failed: %s", e)

        return self._fallback_embed(text)

    def _fallback_embed(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * 384
        for tok in tokens:
            idx = abs(hash(tok)) % 384
            vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
