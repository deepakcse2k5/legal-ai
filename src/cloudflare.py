"""Minimal client for the Cloudflare Workers AI REST API (text generation and embeddings)."""
import httpx

from src.config import settings


class CloudflareError(RuntimeError):
    pass


def run(model: str, payload: dict, timeout: float = 90.0) -> dict:
    if not (settings.cf_account_id and settings.cf_api_token):
        raise CloudflareError("CF_ACCOUNT_ID and CF_API_TOKEN must be set.")
    url = f"https://api.cloudflare.com/client/v4/accounts/{settings.cf_account_id}/ai/run/{model}"
    try:
        r = httpx.post(url, json=payload, headers={"Authorization": f"Bearer {settings.cf_api_token}"}, timeout=timeout)
    except httpx.HTTPError as exc:
        raise CloudflareError(f"Could not reach Cloudflare: {exc}") from exc
    try:
        body = r.json()
    except ValueError:
        raise CloudflareError(f"Cloudflare returned HTTP {r.status_code}.")
    if r.status_code >= 400 or not body.get("success", False):
        errs = "; ".join(str(e.get("message", e)) for e in body.get("errors", [])) or f"HTTP {r.status_code}"
        raise CloudflareError(errs)
    return body.get("result") or {}
