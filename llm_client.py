"""Unified LLM client. Returns (answer_str, raw_response_dict).

Providers: "ollama" (local) or "openai" (Chat Completions API).
Only the *answering* LLM uses this; the judge in accuracy.py is independent.
"""

from __future__ import annotations
import random
import time
from typing import Any

import config

_OPENAI_MAX_RETRIES = 5
_OPENAI_BACKOFF_BASE = 2.0


def get_llm_response(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    provider = config.LLM_PROVIDER
    if provider == "ollama":
        return _ollama_call(prompt)
    if provider == "openai":
        return _openai_call(prompt, json_mode=json_mode)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _ollama_call(prompt: str) -> tuple[str, Any]:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        temperature=config.TEMPERATURE,
        num_predict=config.MAX_TOKENS,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    # Ollama doesn't return token usage — metrics.py uses tiktoken estimation
    return response.content.strip(), {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _openai_call(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    from openai import (
        OpenAI,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
        AuthenticationError,
    )

    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=0, timeout=60.0)

    last_err: Exception | None = None
    for attempt in range(_OPENAI_MAX_RETRIES):
        try:
            kwargs: dict = {
                "model":       config.OPENAI_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": config.TEMPERATURE,
                "max_tokens":  64 if json_mode else config.MAX_TOKENS,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()
            usage = response.usage
            usage_dict = {
                "prompt_tokens":     getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens":      getattr(usage, "total_tokens", 0) if usage else 0,
            }
            return content, {"usage": usage_dict}

        except AuthenticationError as e:
            raise RuntimeError(
                f"OpenAI authentication failed ({e}). Check OPENAI_API_KEY in .env."
            ) from e

        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            last_err = e
            wait = _OPENAI_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            kind = type(e).__name__
            print(f"[llm_client] OpenAI {kind} — retrying in {wait:.1f}s ({attempt + 1}/{_OPENAI_MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError(f"OpenAI call failed after {_OPENAI_MAX_RETRIES} retries: {last_err}")
