"""Small REST client for the Gemini Developer API.

Uses the existing requests dependency; no additional SDK is required.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def generate_text(prompt: str, system_instruction: str = "", *, temperature: float = 0.2,
                  max_output_tokens: int = 2048, timeout: tuple[int, int] = (5, 45)) -> str | None:
    """Return generated text, or None on any provider/API failure."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        response = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if response.status_code >= 400:
            logger.warning("Gemini API returned HTTP %s: %s", response.status_code, response.text[:300])
            return None
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        answer = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        return answer or None
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        logger.exception("Gemini request failed")
        return None
