"""Thin OpenRouter client wrapper (OpenAI-compatible).

All model calls go through here. The key is read from OPENROUTER_API_KEY (loaded
from .env). extra_body carries OpenRouter-specific routing (e.g. provider pinning /
require_parameters) so the experiment's reproducibility controls reach the provider.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def chat(messages: list[dict], slug: str, *, max_tokens: int,
         temperature: float = 0.0, extra_body: dict | None = None,
         client: OpenAI | None = None) -> str:
    """Send a chat completion and return the assistant's text content.

    messages:   list of {"role": ..., "content": ...}
    slug:       OpenRouter model slug (e.g. "openai/gpt-4o")
    max_tokens: required — the value comes from config (models.yaml), so there is no
                second, drifting default here.
    extra_body: OpenRouter-specific fields not in the base OpenAI schema.
    """
    client = client or get_client()
    resp = client.chat.completions.create(
        model=slug,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body or {},
    )
    return resp.choices[0].message.content
