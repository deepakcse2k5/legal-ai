"""Optional dense embeddings (Cloudflare Workers AI). Used to add semantic matching to keyword search."""
import logging
import math

from src import cloudflare
from src.config import settings

log = logging.getLogger("legal-ai")
BATCH = 50


def enabled() -> bool:
    return settings.embeddings == "cloudflare"


def embed(texts: list[str]) -> list[list[float]]:
    """Return one vector per text, or raise CloudflareError."""
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        res = cloudflare.run(settings.cf_embed_model, {"text": texts[i:i + BATCH]})
        data = res.get("data") or []
        if len(data) != len(texts[i:i + BATCH]):
            raise cloudflare.CloudflareError("Unexpected embedding response.")
        out.extend(data)
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
