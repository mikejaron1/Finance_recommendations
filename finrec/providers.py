"""Optional live data lookups — the "search it for the user" layer.

Everything here is best-effort. Each function returns ``None`` on any failure
(no network, rate limited, schema changed) and the caller falls back to the
bundled reference data in :mod:`finrec.lookup`. A planning tool that breaks
because a third-party endpoint is down is worse than one that quietly uses a
two-week-old mortgage rate.

Results are cached on disk so a page refresh doesn't re-hit the network, and
in memory so a single render doesn't either.
"""

from __future__ import annotations

import json
import os
import time
import math
import hashlib
from datetime import date
from dataclasses import dataclass
from pathlib import Path

__all__ = ["current_mortgage_rate", "current_market_data", "clear_cache", "CACHE_DIR"]

CACHE_DIR = Path(os.environ.get("FINREC_CACHE_DIR", Path.home() / ".cache" / "finrec"))
DEFAULT_TTL_SECONDS = 60 * 60 * 12
NETWORK_TIMEOUT = 6.0
FAILURE_BACKOFF_SECONDS = 60
_failed_until: dict[str, float] = {}
_observed_at: dict[str, str] = {}

# Used when the network is unavailable. Update alongside the reference year.
FALLBACK_MORTGAGE_RATE_30Y = 0.0665
FALLBACK_MORTGAGE_RATE_15Y = 0.0585


@dataclass
class Fetched:
    value: float
    source: str
    fetched_at: float
    live: bool
    observation_date: str | None = None


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str, ttl: float) -> dict | None:
    path = _cache_path(key)
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return None
        stamp = payload.get("fetched_at")
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= time.time() - stamp <= ttl:
            return None
        value = payload.get("value")
        if key.startswith("mortgage_"):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 0.5:
                return None
        elif not isinstance(value, dict) or not value or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in value.values()):
            return None
        observed = payload.get("observation_date")
        if observed is not None and date.fromisoformat(observed) > date.today():
            return None
        return payload
    except Exception:
        return None


def _write_cache(key: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(key).write_text(json.dumps(payload))
    except Exception:
        pass


def clear_cache() -> None:
    _failed_until.clear()
    _observed_at.clear()
    try:
        for path in CACHE_DIR.glob("*.json"):
            path.unlink()
    except Exception:
        pass


def _fetch_fred_series(series_id: str) -> float | None:
    """Latest observation of a FRED series via the public CSV endpoint.

    Deliberately uses the keyless graph endpoint: requiring users to register
    for an API key to see a mortgage rate would defeat the purpose.
    """
    try:
        import requests
    except ImportError:
        return None

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        response = requests.get(url, timeout=NETWORK_TIMEOUT)
        response.raise_for_status()
        for line in reversed(response.text.strip().splitlines()):
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                value = float(parts[-1])
                observed = date.fromisoformat(parts[0])
                if not math.isfinite(value) or value <= 0 or observed > date.today():
                    continue
                _observed_at[series_id] = observed.isoformat()
                return value
            except (ValueError, TypeError):
                continue
    except Exception:
        return None
    return None


def current_mortgage_rate(term_years: int = 30, *, ttl: float = DEFAULT_TTL_SECONDS) -> float | None:
    """Current average US mortgage rate as a decimal (0.0665 = 6.65%)."""
    series = "MORTGAGE30US" if term_years >= 20 else "MORTGAGE15US"
    key = f"mortgage_{series}"

    cached = _read_cache(key, ttl)
    if cached:
        return cached["value"]
    if _failed_until.get(key, 0) > time.time():
        return None

    percent = _fetch_fred_series(series)
    if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 < percent <= 50:
        _failed_until[key] = time.time() + FAILURE_BACKOFF_SECONDS
        return None

    value = round(percent / 100.0, 5)
    _failed_until.pop(key, None)
    _write_cache(key, {"value": value, "source": f"FRED {series}", "fetched_at": time.time(),
                      "observation_date": _observed_at.get(series)})
    return value


def mortgage_rate_or_default(term_years: int = 30, *, live: bool = True) -> tuple[float, bool]:
    """``(rate, is_live)`` — always returns something usable."""
    rate = current_mortgage_rate(term_years) if live else None
    if rate is not None:
        return rate, True
    return (FALLBACK_MORTGAGE_RATE_30Y if term_years >= 20 else FALLBACK_MORTGAGE_RATE_15Y), False


def current_market_data(tickers: tuple[str, ...] = ("^GSPC",), *, ttl: float = DEFAULT_TTL_SECONDS) -> dict:
    """Latest close for a handful of tickers, if yfinance is installed."""
    key = "market_" + hashlib.sha256(json.dumps(tickers).encode()).hexdigest()[:24]
    cached = _read_cache(key, ttl)
    if cached:
        return cached["value"]
    if _failed_until.get(key, 0) > time.time():
        return {}

    try:
        import yfinance as yf
    except ImportError:
        _failed_until[key] = time.time() + FAILURE_BACKOFF_SECONDS
        return {}

    out: dict[str, float] = {}
    try:
        for ticker in tickers:
            history = yf.Ticker(ticker).history(period="5d")
            if len(history):
                value = float(history["Close"].iloc[-1])
                if math.isfinite(value) and value > 0:
                    out[ticker] = value
                    _observed_at[ticker] = str(history.index[-1].date())
    except Exception:
        _failed_until[key] = time.time() + FAILURE_BACKOFF_SECONDS
        return {}

    if out:
        _write_cache(key, {"value": out, "source": "yfinance", "fetched_at": time.time(),
                          "observation_dates": {t: _observed_at.get(t) for t in out}})
    else:
        _failed_until[key] = time.time() + FAILURE_BACKOFF_SECONDS
    return out
