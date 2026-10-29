"""Keyword retrieval (BM25). Strong for legal text, where exact terms matter, and needs no model download."""
import math
import re
from collections import Counter

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
