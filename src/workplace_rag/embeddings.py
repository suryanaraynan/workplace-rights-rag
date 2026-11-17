"""Text -> vector, using a small local sentence-transformers model.

Why local embeddings when the LLM is hosted?
- Free and private: the 463 chunks are embedded once, on your machine.
- Small: all-MiniLM-L6-v2 is ~90 MB and fast on a laptop CPU.
- Swappable: the model name is in config.yaml (step 3 compares models).
"""
from __future__ import annotations


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix = query_prefix  # some models (e.g. bge) want "query: "-style prefixes
        self._model = None

    @property
    def model(self):
        if self._model is None:  # lazy: importing torch is slow, only do it when needed
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 100)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].tolist()
