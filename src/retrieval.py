"""Keyword retrieval (BM25). Strong for legal text, where exact terms matter, and needs no model download."""
import math
import re
from collections import Counter

from src import embeddings
from src.loader import Chunk

_STOP = set("a an the of to in on and or is are be by for with at as it its this that what which who how when does do can may shall any if from not under than then there their they".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


class Retriever:
    """BM25 with the non-negative IDF variant, so it also behaves on very small documents."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(t.values()) for t in self._tf]
        self._avg = (sum(self._len) / len(chunks)) if chunks else 1.0
        df: Counter = Counter()
        for t in self._tf:
            df.update(t.keys())
        n = len(chunks)
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def _score(self, i: int, query: list[str]) -> float:
        tf, dl, s = self._tf[i], self._len[i], 0.0
        for w in query:
            f = tf.get(w, 0)
            if f:
                s += self._idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avg))
        return s

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        q = tokenize(question)
        if not q or not self.chunks:
            return []
        scored = sorted(((self._score(i, q), i) for i in range(len(self.chunks))), reverse=True)
        return [self.chunks[i] for s, i in scored[:k] if s > 0]


def _rank(scores: list[float]) -> dict[int, int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {i: r for r, i in enumerate(order, start=1) if scores[i] > 0}


class HybridRetriever(Retriever):
    """Keyword (BM25) plus semantic (embeddings), merged with reciprocal rank fusion.

    Keyword search is precise on exact legal terms; embeddings catch paraphrases such as
    "how do I get out of the contract" vs "termination for convenience". If embeddings are
    unavailable the retriever silently behaves as plain BM25.
    """

    def __init__(self, chunks: list[Chunk]):
        super().__init__(chunks)
        self.vectors: list[list[float]] | None = None
        if chunks and embeddings.enabled():
            try:
                self.vectors = embeddings.embed([c.text for c in chunks])
            except Exception as exc:  # network, quota, bad token: fall back to keyword search
                import logging

                logging.getLogger("legal-ai").warning("Embeddings unavailable, using keyword search only: %s", exc)

    @property
    def semantic(self) -> bool:
        return self.vectors is not None

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        if not self.semantic:
            return super().search(question, k)
        q = tokenize(question)
        kw = [self._score(i, q) for i in range(len(self.chunks))] if q else [0.0] * len(self.chunks)
        try:
            qv = embeddings.embed([question])[0]
            sem = [embeddings.cosine(qv, v) for v in self.vectors]
        except Exception:
            return super().search(question, k)
        r_kw, r_sem = _rank(kw), _rank(sem)
        fused = {i: 1 / (60 + r_kw[i]) if i in r_kw else 0.0 for i in range(len(self.chunks))}
        for i, r in r_sem.items():
            if r <= 20:                                   # only strong semantic matches vote
                fused[i] += 1 / (60 + r)
        ranked = sorted(fused, key=fused.get, reverse=True)
        return [self.chunks[i] for i in ranked[:k] if fused[i] > 0]
