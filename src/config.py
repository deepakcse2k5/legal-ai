"""Settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")          # "ollama", "groq" or "cloudflare"
    cf_account_id: str = os.getenv("CF_ACCOUNT_ID", "")
    cf_api_token: str = os.getenv("CF_API_TOKEN", "")
    cf_model: str = os.getenv("CF_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    cf_embed_model: str = os.getenv("CF_EMBED_MODEL", "@cf/baai/bge-base-en-v1.5")
    embeddings: str = os.getenv("EMBEDDINGS", "none")                 # "cloudflare" or "none" (keyword search only)
    api_key: str = os.getenv("API_KEY", "")                           # if set, callers must send X-API-Key
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_answer_tokens: int = _int("MAX_ANSWER_TOKENS", 500)
    max_summary_tokens: int = _int("MAX_SUMMARY_TOKENS", 800)
    context_tokens: int = _int("CONTEXT_TOKENS", 8192)               # Ollama num_ctx
    top_k: int = _int("TOP_K", 4)                                    # passages per question
    chunk_chars: int = _int("CHUNK_CHARS", 1200)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 20)
    max_pages: int = _int("MAX_PAGES", 300)
    session_ttl_seconds: int = _int("SESSION_TTL_SECONDS", 3600)
    max_sessions: int = _int("MAX_SESSIONS", 100)
    cors_origins: tuple = tuple(o for o in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if o)


settings = Settings()
