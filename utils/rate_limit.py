"""In-memory rate limiter (token bucket), client-IP helper, and Flask wiring.

Domain:    Utils — Rate Limiting
Context:
  - This project throttles in two different directions — do not conflate them:
      1. ``utils/network.py::yf_throttle`` — *outbound* throttle protecting the
         Yahoo Finance upstream (see ADR 0005).
      2. This module — *inbound* throttle protecting the Flask process from
         abusive or buggy clients.
  - The inbound limiter is dependency-free on purpose: a dict + lock is
    adequate for a single-machine personal project.
  - TRADEOFF: state is per-process. Under ``gunicorn --workers N`` the
    effective budget is ``N × max_calls`` because every worker counts
    independently. A shared backend (Redis) would fix that at the cost of a
    runtime dependency this project deliberately avoids.
Contracts:
  - rate_limit(key, max_calls, window_sec) -> tuple[bool, int]
  - parse_rate_limit_spec(spec) -> tuple[int, int]
  - client_ip() -> str
  - install(app) -> bool   (registers the global before_request throttle)
Dependencies UPWARD:
  - flask (request, jsonify)
Dependencies DOWNWARD:
  - app.py (install), routes/* (rate_limit)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time

from flask import jsonify, request

logger = logging.getLogger(__name__)

_rate_buckets: dict[str, list[float]] = {}
_rate_lock = threading.Lock()

# Bound on the bucket map. Without it, forged client IPs (see client_ip) grow
# the dict without limit — a memory-exhaustion vector on a single-process app.
_MAX_BUCKETS = 10_000

# Default global budget per client IP. Kept identical to the previously used
# flask-limiter default so removing that dependency changed no behaviour.
DEFAULT_RATE_LIMIT_SPEC = "120 per minute"

# DOMAIN: window granularity is intentionally coarse (minute/hour/day) —
# sub-minute windows would reject normal HTMX tab fan-out on page load.
_WINDOW_SECONDS = {
    "second": 1,
    "sec": 1,
    "s": 1,
    "minute": 60,
    "min": 60,
    "m": 60,
    "hour": 3600,
    "hr": 3600,
    "h": 3600,
    "day": 86400,
    "d": 86400,
}

# Accepts "120 per minute", "120/minute", "120 per 1 minute".
_SPEC_RE = re.compile(r"^(\d+)\s*(?:/|per)\s*(\d+\s+)?([a-z]+)$", re.IGNORECASE)


def _prune_buckets(now: float, window_sec: int) -> None:
    """Keep ``_rate_buckets`` under ``_MAX_BUCKETS``. Caller holds ``_rate_lock``.

    Emergency-shed semantics: when at cap we first drop fully-expired
    buckets, then evict the least-recently-hit ones. Buckets created under a
    longer window may be evicted early — acceptable under an over-cap
    emergency (worst case: that client gets a fresh budget). Evicts down to
    one slot below the cap so the caller's insert stays within budget.
    """
    if len(_rate_buckets) < _MAX_BUCKETS:
        return
    for key in list(_rate_buckets):
        live = [t for t in _rate_buckets[key] if (now - t) < window_sec]
        if live:
            _rate_buckets[key] = live
        else:
            del _rate_buckets[key]
    while len(_rate_buckets) >= _MAX_BUCKETS:
        oldest_key = min(_rate_buckets, key=lambda k: _rate_buckets[k][-1])
        del _rate_buckets[oldest_key]


def rate_limit(key: str, max_calls: int, window_sec: int) -> tuple[bool, int]:
    """Allow up to ``max_calls`` per ``window_sec`` per key.

    Returns ``(allowed, retry_after_seconds)``. When throttled, retry_after
    is the number of seconds until the oldest call in the window expires.
    """
    now = time.monotonic()
    with _rate_lock:
        _prune_buckets(now, window_sec)
        bucket = _rate_buckets.get(key, [])
        # Drop expired entries
        bucket = [t for t in bucket if (now - t) < window_sec]
        if len(bucket) >= max_calls:
            retry = int(window_sec - (now - bucket[0])) + 1
            _rate_buckets[key] = bucket
            return False, max(1, retry)
        bucket.append(now)
        _rate_buckets[key] = bucket
    return True, 0


def parse_rate_limit_spec(spec: str) -> tuple[int, int]:
    """Parse ``"<n> per <unit>"`` / ``"<n>/<unit>"`` into ``(max_calls, window_sec)``.

    Raises ``ValueError`` on unparseable input so callers can fall back to the
    default and log — never silently throttle with a nonsense budget.
    """
    match = _SPEC_RE.match((spec or "").strip())
    if not match:
        raise ValueError(f"unparseable rate-limit spec: {spec!r}")

    max_calls = int(match.group(1))
    multiplier = int(match.group(2)) if match.group(2) else 1
    unit = match.group(3).lower()

    if max_calls <= 0 or multiplier <= 0 or unit not in _WINDOW_SECONDS:
        raise ValueError(f"unparseable rate-limit spec: {spec!r}")
    return max_calls, _WINDOW_SECONDS[unit] * multiplier


def _trust_forwarded_header() -> bool:
    """Whether X-Forwarded-For may be used as the throttle key.

    SECURITY: direct listeners (``app.run`` / bare gunicorn) must NOT trust
    this header — clients can forge it and rotate it per request, resetting
    every per-IP budget. Only enable this when the app sits behind a proxy
    that overwrites the header.
    """
    return os.environ.get("TRUST_X_FORWARDED_FOR", "").strip().lower() in ("1", "true", "yes")


def client_ip() -> str:
    """Return the client IP used as the throttle key."""
    if _trust_forwarded_header():
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def install(app) -> bool:
    """Register the global per-IP request throttle on a Flask app.

    Returns ``True`` when the throttle was installed, ``False`` when it was
    disabled via ``RATE_LIMIT_DISABLED``.

    WHY: this used to be delegated to ``flask-limiter``, which signals 429 by
    raising a Werkzeug ``HTTPException``. That bypassed the unified JSON
    envelope in ``utils/api_errors.py`` — its catch-all handler re-raises
    ``HTTPException`` unchanged — so ``/api/*`` clients got an HTML error page
    instead of ``{"status": "error", "code": "rate_limited", ...}``. Emitting
    the envelope here keeps one error contract shared with the per-endpoint
    checks in ``routes/``.
    """
    if os.environ.get("RATE_LIMIT_DISABLED", "").strip() in ("1", "true", "yes"):
        logger.info("Rate limiter disabled via RATE_LIMIT_DISABLED")
        return False

    spec = os.environ.get("RATE_LIMIT_DEFAULT", DEFAULT_RATE_LIMIT_SPEC)
    try:
        max_calls, window_sec = parse_rate_limit_spec(spec)
    except ValueError:
        max_calls, window_sec = parse_rate_limit_spec(DEFAULT_RATE_LIMIT_SPEC)
        logger.warning("Invalid RATE_LIMIT_DEFAULT=%r — falling back to %r", spec, DEFAULT_RATE_LIMIT_SPEC)

    @app.before_request
    def _throttle_global():
        # Static assets are served by Flask's own route; counting them would
        # punish a single page load with dozens of requests.
        if request.endpoint == "static":
            return None

        allowed, retry_after = rate_limit(f"global:{client_ip()}", max_calls, window_sec)
        if allowed:
            return None

        logger.warning("Rate limit exceeded for %s (%s per %ss)", client_ip(), max_calls, window_sec)
        payload = {
            "status": "error",
            "code": "rate_limited",
            "message": f"Too many requests. Retry after {retry_after}s.",
            "details": {"retry_after": retry_after},
        }
        return jsonify(payload), 429, {"Retry-After": str(retry_after)}

    logger.info("Rate limiter enabled: default=%s per %ss", max_calls, window_sec)
    return True
