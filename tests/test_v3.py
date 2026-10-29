import io

import httpx
import pytest

from src import cloudflare, embeddings, llm, service
from src.app import app
from src.config import settings
from src.loader import Chunk
from src.retrieval import HybridRetriever
from tests.conftest import make_pdf


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr("src.app.settings", settings.__class__(api_key="s3cret"))


def test_health_is_open_but_other_calls_need_the_key(client, with_key):
    assert client.get("/health").status_code == 200
    assert client.get("/drafts").status_code == 401
    assert client.get("/drafts", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/drafts", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_no_key_configured_means_open_for_local_dev(client):
    assert client.get("/drafts").status_code == 200


def test_cloudflare_provider_builds_request_and_parses_response(monkeypatch):
    monkeypatch.setattr(cloudflare, "settings", settings.__class__(cf_account_id="acc", cf_api_token="tok"))
    monkeypatch.setattr(llm, "settings", settings.__class__(llm_provider="cloudflare", cf_model="@cf/test/model"))
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"success": True, "result": {"response": "Hello <think>x</think>world [p. 2]"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = llm.generate("sys", "user", 100)
    assert out == "Hello world [p. 2]"
    assert seen["url"].endswith("/accounts/acc/ai/run/@cf/test/model")
    assert seen["headers"]["Authorization"] == "Bearer tok"
    assert seen["json"]["messages"][0] == {"role": "system", "content": "sys"}


def test_cloudflare_error_becomes_llm_error(monkeypatch):
    monkeypatch.setattr(cloudflare, "settings", settings.__class__(cf_account_id="acc", cf_api_token="tok"))
    monkeypatch.setattr(llm, "settings", settings.__class__(llm_provider="cloudflare"))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(429, json={"success": False, "errors": [{"message": "daily allocation exceeded"}]}))
    with pytest.raises(llm.LLMError, match="daily allocation exceeded"):
        llm.generate("s", "u", 10)


def test_hybrid_retriever_finds_paraphrase_that_keywords_miss(monkeypatch):
    chunks = [Chunk(1, 1, "Either party may end this agreement on sixty days notice."),
              Chunk(2, 1, "Fees are payable monthly in advance."),
              Chunk(3, 2, "This agreement is governed by Indian law.")]
    vec = {chunks[0].text: [1, 0], chunks[1].text: [0, 1], chunks[2].text: [-1, 0], "how do I get out of the contract": [0.9, 0.1]}
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "embed", lambda texts: [vec[t] for t in texts])
    r = HybridRetriever(chunks)
    assert r.semantic
    assert r.search("how do I get out of the contract", 1)[0].id == 1      # no shared keyword at all


def test_hybrid_falls_back_to_keywords_when_embeddings_fail(monkeypatch):
    monkeypatch.setattr(embeddings, "enabled", lambda: True)

    def boom(texts):
        raise cloudflare.CloudflareError("down")

    monkeypatch.setattr(embeddings, "embed", boom)
    r = HybridRetriever([Chunk(1, 1, "Payment within thirty days."), Chunk(2, 1, "Governing law is India.")])
    assert not r.semantic
    assert r.search("payment days", 1)[0].id == 1


def test_upload_reports_semantic_flag(client):
    r = client.post("/upload", files={"file": ("a.pdf", io.BytesIO(make_pdf()), "application/pdf")})
    assert r.json()["semantic_search"] is False
