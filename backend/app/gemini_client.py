"""Small REST client for the Gemini Developer API.

Uses the existing requests dependency; no additional SDK is required.

Environment variables (all optional):
    GEMINI_API_KEY       your key (required to use Gemini)
    GEMINI_MODEL         default "gemini-3.8-flash"
    GEMINI_MAX_TIMEOUT   upper bound in seconds for any single call (default 120). Callers choose
                         their own timeout; this only stops a caller from asking for too much.
    GEMINI_THINKING      "auto" (default), "off", "minimal", "low", "medium", "high" or "default".
                         auto = minimal thinking for short structured outputs (<= 3000 tokens, e.g. the
                         planner and verifier) and the model default for longer answers. Thinking tokens
                         count against maxOutputTokens, so too much thinking can leave an empty reply.
    GEMINI_BASE_URL      override the endpoint (tests, proxies)
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _base_url() -> str:
    return os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")


def _max_timeout() -> float:
    try:
        return max(5.0, float(os.getenv("GEMINI_MAX_TIMEOUT", "120")))
    except ValueError:
        return 120.0


def _thinking_config(model: str, max_output_tokens: int) -> dict | None:
    """Map our thinking mode to the parameter this model family understands (None = leave default)."""
    mode = os.getenv("GEMINI_THINKING", "auto").strip().lower()
    if mode == "auto":
        mode = "minimal" if max_output_tokens <= 3000 else "default"
    if mode in ("default", ""):
        return None
    name = model.lower()
    if "2.5" in name:  # Gemini 2.5 uses a token budget
        budget = {"off": 0, "minimal": 0, "low": 1024, "medium": 4096, "high": 8192}.get(mode)
        if budget is None:
            return None
        return {"thinkingBudget": max(budget, 128) if "pro" in name else budget}
    if re.match(r"gemini-([3-9]|\d{2})", name):  # Gemini 3 and later use a level
        level = {"off": "minimal", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high"}.get(mode)
        if level == "minimal" and "pro" in name:
            level = "low"  # Pro models can't turn thinking fully off
        return {"thinkingLevel": level} if level else None
    return None


def _retry_after(response: requests.Response) -> float:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        pass
    match = re.search(r"retry(?:Delay)?\D{0,12}(\d+(?:\.\d+)?)\s*s", response.text or "", re.I)
    return float(match.group(1)) if match else 0.0


def generate_text(
        prompt: str,
        system_instruction: str = "",
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        timeout: tuple[int, int] = (5, 20),
) -> str | None:
    """Return generated text, or None on any provider/API failure.

    timeout = (connect seconds, total seconds). The total is a budget for the whole call including
    retries, honoured up to GEMINI_MAX_TIMEOUT.
    """

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        logger.warning("Gemini API key is missing.")
        return None

    connect_timeout = min(max(int(timeout[0]), 1), 10)
    budget = min(max(float(timeout[1]), 1.0), _max_timeout())
    started = time.monotonic()
    deadline = started + budget

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()
    url = f"{_base_url()}/models/{model}:generateContent"

    generation_config: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    thinking = _thinking_config(model, max_output_tokens)
    if thinking:
        generation_config["thinkingConfig"] = thinking

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    raised_tokens = False
    best_text: str | None = None
    errors = 0

    for attempt in range(6):  # bounded: retries, a thinking fallback and one token-budget increase
        remaining = deadline - time.monotonic()
        if attempt > 0 and remaining < 2:  # the first request is always sent
            logger.warning("Gemini call ran out of its %.0fs budget.", budget)
            return best_text

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=(connect_timeout, remaining))
        except requests.Timeout:
            logger.warning(
                "Gemini request timed out after %.0f seconds (budget %.0fs, model %s). "
                "Raise the caller's timeout or GEMINI_MAX_TIMEOUT.",
                time.monotonic() - started, budget, model,
                )
            return best_text
        except requests.RequestException as exc:
            logger.warning("Gemini network request failed: %s", exc)
            return best_text

        code = response.status_code

        if code == 400 and "thinkingConfig" in generation_config and "think" in response.text.lower():
            logger.info("Gemini model %s rejected the thinking setting; retrying without it.", model)
            generation_config.pop("thinkingConfig")
            continue

        if code in (429, 500, 502, 503, 504):
            errors += 1
            wait = _retry_after(response) if code == 429 else 0.0
            wait = wait or min(2.0 * errors, 6.0)
            if errors >= 3 or wait > min(20.0, remaining - 3):
                logger.warning("Gemini returned HTTP %s and will not be retried (wait %.0fs): %s",
                               code, wait, response.text[:200])
                return best_text
            logger.warning("Gemini returned HTTP %s. Retrying in %.1fs.", code, wait)
            time.sleep(wait)
            continue

        if code >= 400:
            logger.warning("Gemini API returned HTTP %s: %s", code, response.text[:300])
            return best_text

        try:
            data = response.json()
        except ValueError as exc:
            logger.warning("Could not parse Gemini response: %s", exc)
            return best_text

        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            logger.warning("Gemini blocked the prompt (%s).", block)
            return best_text

        candidates = data.get("candidates") or []
        if not candidates:
            logger.warning("Gemini returned no candidates.")
            return best_text

        candidate = candidates[0]
        finish = candidate.get("finishReason", "")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and not part.get("thought")
        ).strip()

        if finish == "MAX_TOKENS" and not raised_tokens and max_output_tokens < 16384:
            # Thinking tokens share the output budget, so the answer was cut short or came back empty.
            raised_tokens = True
            best_text = text or best_text
            max_output_tokens = min(max_output_tokens * 3, 16384)
            generation_config["maxOutputTokens"] = max_output_tokens
            logger.warning("Gemini hit MAX_TOKENS (%s chars so far). Retrying with %s tokens.",
                           len(text), max_output_tokens)
            continue

        if not text:
            logger.warning("Gemini returned no text (finishReason=%s).", finish or "unknown")
            return best_text

        if finish == "MAX_TOKENS":
            logger.warning("Gemini output was truncated at the token limit.")

        logger.info("Gemini %s ok in %.1fs (%d chars, finish=%s).", model, time.monotonic() - started,
                    len(text), finish or "unknown")
        return text

    return best_text
