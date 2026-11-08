"""Password hashing, session tokens and a tiny in-memory rate limiter (stdlib only)."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict, deque

SESSION_DAYS = int(os.getenv("SESSION_DAYS", "30"))
MIN_PASSWORD = 8
MAX_PASSWORD = 128
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")

# scrypt parameters (OWASP-acceptable: N=2^14, r=8, p=1)
_N, _R, _P = 2**14, 8, 1


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        raise ValueError("Enter a valid email address.")
    return email


def validate_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    if len(password) > MAX_PASSWORD:
        raise ValueError(f"Password must be at most {MAX_PASSWORD} characters.")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        probe = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(probe, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


# Used to spend the same time on unknown emails, so login timing doesn't reveal which emails exist.
DUMMY_HASH = hash_password("not-a-real-password")


def new_token() -> tuple[str, str]:
    """Return (token to give the client, sha256 hash to store). Only the hash is kept server-side."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class RateLimiter:
    """Sliding-window limiter. In-memory, so it is per server process (fine for one worker)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > window_seconds:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        if len(self._hits) > 20000:  # keep memory bounded
            for stale in [k for k, v in self._hits.items() if not v or now - v[-1] > window_seconds]:
                self._hits.pop(stale, None)
        return True
