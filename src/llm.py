"""LLM access behind one function. Providers: Cloudflare Workers AI, Groq, or local Ollama."""
import re

from src import cloudflare
from src.config import settings


class LLMError(RuntimeError):
    pass


def _strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(system: str, user: str, max_tokens: int) -> str:
    try:
        if settings.llm_provider == "groq":
            from groq import Groq

            if not settings.groq_api_key:
                raise LLMError("GROQ_API_KEY is not set.")
            out = Groq(api_key=settings.groq_api_key).chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=settings.temperature,
                max_tokens=max_tokens,
            )
            text = out.choices[0].message.content or ""
        elif settings.llm_provider == "cloudflare":
            text = cloudflare.run(
                settings.cf_model,
                {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                 "max_tokens": max_tokens, "temperature": settings.temperature},
            ).get("response") or ""
        else:
            import ollama

            out = ollama.chat(
                model=settings.ollama_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": settings.temperature, "num_predict": max_tokens, "num_ctx": settings.context_tokens},
            )
            text = out["message"]["content"] or ""
    except LLMError:
        raise
    except Exception as exc:  # network, model missing, quota ...
        raise LLMError(f"The language model call failed: {exc}") from exc
    text = _strip_reasoning(text)
    if not text:
        raise LLMError("The language model returned an empty answer.")
    return text
