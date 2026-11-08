
"""Small REST client for the Gemini Developer API.

Uses the existing requests dependency; no additional SDK is required.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def generate_text(
        prompt: str,
        system_instruction: str = "",
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        timeout: tuple[int, int] = (5, 20),
) -> str | None:
    """Return generated text, or None on any provider/API failure."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        logger.warning("Gemini API key is missing.")
        return None

    # Limit timeouts even if another module requests a longer timeout.
    connect_timeout = min(max(int(timeout[0]), 1), 5)
    read_timeout = min(max(int(timeout[1]), 1), 20)
    safe_timeout = (connect_timeout, read_timeout)

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    for attempt in range(2):
        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=safe_timeout,
            )

            if response.status_code == 503 and attempt == 0:
                logger.warning("Gemini returned HTTP 503. Retrying once.")
                time.sleep(1)
                continue

            if response.status_code >= 400:
                logger.warning(
                    "Gemini API returned HTTP %s: %s",
                    response.status_code,
                    response.text[:300],
                )
                return None

            data = response.json()
            candidates = data.get("candidates", [])

            if not candidates:
                logger.warning("Gemini returned no candidates.")
                return None

            parts = candidates[0].get("content", {}).get("parts", [])

            answer = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            ).strip()

            return answer or None

        except requests.Timeout:
            logger.warning(
                "Gemini request timed out after %s seconds.",
                read_timeout,
            )
            return None

        except requests.RequestException as exc:
            logger.warning("Gemini network request failed: %s", exc)
            return None

        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Could not parse Gemini response: %s", exc)
            return None

    return None
