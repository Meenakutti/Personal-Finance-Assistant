"""
Market Data Integration: Fetch real-time market data from various sources.
30-minute TTL cache, exponential backoff retries, stale-cache and mock fallbacks.
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)


# Minimal mock data used only when no cache exists and API is completely unavailable
_MOCK_INDICES: Dict[str, Dict[str, float]] = {
    "S&P 500":      {"price": 5200.0, "change": 0.0, "change_percent": 0.0},
    "Dow Jones":    {"price": 42000.0, "change": 0.0, "change_percent": 0.0},
    "Nasdaq":       {"price": 16800.0, "change": 0.0, "change_percent": 0.0},
    "Russell 2000": {"price": 2100.0,  "change": 0.0, "change_percent": 0.0},
    "VIX":          {"price": 15.0,    "change": 0.0, "change_percent": 0.0},
}

_MOCK_SECTORS: Dict[str, float] = {
    "Technology": 0.0, "Healthcare": 0.0, "Financials": 0.0,
    "Energy": 0.0, "Industrials": 0.0, "Consumer Discretionary": 0.0,
    "Materials": 0.0, "Real Estate": 0.0, "Utilities": 0.0,
    "Communication Services": 0.0,
}

_SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Energy": "XLE", "Industrials": "XLI", "Consumer Discretionary": "XLY",
    "Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
    "Communication Services": "XLC",
}

_MAJOR_INDICES: Dict[str, str] = {
    "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC",
    "Russell 2000": "^RUT", "VIX": "^VIX",
}


class MarketDataProvider:
    """
    Integrates with yFinance for market data.

    Caching strategy:
    - Live cache: 30-minute TTL; subsequent requests within the window are served instantly.
    - Stale cache: last successfully fetched value kept indefinitely as a fallback.
    - Mock data: hardcoded approximations used only when stale cache is also empty.

    Retry strategy: up to 3 attempts with exponential backoff (1 s, 2 s, 4 s).
    """

    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self._live_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._stale_cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._max_retries = 3

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _is_cache_valid(self, key: str) -> bool:
        return key in self._cache_expiry and datetime.now() < self._cache_expiry[key]

    def _cache_data(self, key: str, data: Any) -> None:
        self._live_cache[key] = data
        self._cache_expiry[key] = datetime.now() + self._cache_ttl
        self._stale_cache[key] = data  # always update stale reference

    def _get_fallback(self, key: str, mock_data: Any) -> Any:
        """Return stale cache when available, otherwise mock data."""
        if key in self._stale_cache:
            _tracer.warn("stale_cache_used", key=key)
            return self._stale_cache[key]
        _tracer.warn("mock_data_used", key=key)
        return mock_data

    def clear_cache(self) -> None:
        """Clear live cache; stale cache is preserved for fallback."""
        self._live_cache.clear()
        self._cache_expiry.clear()

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _fetch_with_retry(self, fetch_fn, *args, **kwargs) -> Any:
        """
        Call fetch_fn with exponential backoff on transient failures.
        Delays: 1 s → 2 s → 4 s before each retry.
        """
        last_exc: Optional[Exception] = None
        fn_name = fetch_fn.__name__
        for attempt in range(self._max_retries):
            try:
                t0 = time.perf_counter()
                result = fetch_fn(*args, **kwargs)
                _tracer.timing("fetch_success", time.perf_counter() - t0,
                               fn=fn_name, attempt=attempt + 1)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    delay = 2 ** attempt  # 1, 2, 4 seconds
                    _tracer.warn("fetch_retry", fn=fn_name,
                                 attempt=attempt + 1, delay_s=delay, error=str(exc))
                    time.sleep(delay)
                else:
                    _tracer.error("fetch_all_retries_failed", fn=fn_name,
                                  attempts=self._max_retries, error=str(exc))
        raise last_exc  # type: ignore[misc]

    # ── Stock quote ───────────────────────────────────────────────────────────

    def _do_fetch_quote(self, symbol: str) -> Dict[str, Any]:
        info = yf.Ticker(symbol).info
        return {
            "symbol": symbol,
            "price": float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0),
            "change": float(info.get("regularMarketChange") or 0.0),
            "change_percent": float(info.get("regularMarketChangePercent") or 0.0),
            "timestamp": datetime.now().isoformat(),
        }

    def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current stock quote with caching, retry, and fallback."""
        key = f"quote_{symbol}"
        if self._is_cache_valid(key):
            _tracer.detail("cache_hit", key=key)
            return self._live_cache[key]
        _tracer.step("fetch_quote", symbol=symbol)
        try:
            data = self._fetch_with_retry(self._do_fetch_quote, symbol)
            self._cache_data(key, data)
            _tracer.detail("quote_cached", symbol=symbol, price=data.get("price"))
            return data
        except Exception as e:
            _tracer.error("quote_fetch_failed", symbol=symbol, error=str(e))
            return self._get_fallback(key, {
                "symbol": symbol, "price": 0.0, "change": 0.0,
                "change_percent": 0.0, "timestamp": datetime.now().isoformat(),
            })

    # ── Stock history ─────────────────────────────────────────────────────────

    def _do_fetch_history(self, symbol: str, days: int) -> List[Dict]:
        end = datetime.now()
        start = end - timedelta(days=days)
        hist = yf.Ticker(symbol).history(start=start, end=end)
        return [
            {
                "date": str(date),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
            for date, row in hist.iterrows()
        ]

    def get_stock_history(self, symbol: str, days: int = 30) -> Optional[List[Dict]]:
        """Get historical OHLCV data with caching, retry, and fallback."""
        key = f"history_{symbol}_{days}"
        if self._is_cache_valid(key):
            _tracer.detail("cache_hit", key=key)
            return self._live_cache[key]
        _tracer.step("fetch_history", symbol=symbol, days=days)
        try:
            data = self._fetch_with_retry(self._do_fetch_history, symbol, days)
            self._cache_data(key, data)
            _tracer.detail("history_cached", symbol=symbol, rows=len(data))
            return data
        except Exception as e:
            _tracer.error("history_fetch_failed", symbol=symbol, error=str(e))
            return self._get_fallback(key, [])

    # ── Sector performance ────────────────────────────────────────────────────

    def _do_fetch_sectors(self) -> Dict[str, float]:
        sectors: Dict[str, float] = {}
        for name, ticker in _SECTOR_ETFS.items():
            try:
                info = yf.Ticker(ticker).info
                sectors[name] = float(info.get("regularMarketChangePercent") or 0.0)
            except Exception:
                sectors[name] = 0.0
        return sectors

    def get_sector_performance(self) -> Optional[Dict[str, float]]:
        """Get sector ETF performance with caching, retry, and fallback."""
        key = "sector_performance"
        if self._is_cache_valid(key):
            _tracer.detail("cache_hit", key=key)
            return self._live_cache[key]
        _tracer.step("fetch_sectors")
        try:
            data = self._fetch_with_retry(self._do_fetch_sectors)
            self._cache_data(key, data)
            _tracer.detail("sectors_cached", count=len(data))
            return data
        except Exception as e:
            _tracer.error("sector_fetch_failed", error=str(e))
            return self._get_fallback(key, dict(_MOCK_SECTORS))

    # ── Market indices ────────────────────────────────────────────────────────

    def _do_fetch_indices(self) -> Dict[str, Dict[str, float]]:
        indices: Dict[str, Dict[str, float]] = {}
        for name, ticker in _MAJOR_INDICES.items():
            try:
                fast = yf.Ticker(ticker).fast_info
                price = float(getattr(fast, "last_price", None) or 0.0)
                prev_close = float(getattr(fast, "previous_close", None) or price)
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0.0
                indices[name] = {
                    "price": price,
                    "change": round(change, 2),
                    "change_percent": round(change_pct, 4),
                }
            except Exception:
                indices[name] = {"price": 0.0, "change": 0.0, "change_percent": 0.0}
        return indices

    def get_market_indices(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get major index data with caching, retry, and fallback."""
        key = "market_indices"
        if self._is_cache_valid(key):
            _tracer.detail("cache_hit", key=key)
            return self._live_cache[key]
        _tracer.step("fetch_indices")
        try:
            data = self._fetch_with_retry(self._do_fetch_indices)
            self._cache_data(key, data)
            _tracer.detail("indices_cached", count=len(data))
            return data
        except Exception as e:
            _tracer.error("indices_fetch_failed", error=str(e))
            return self._get_fallback(key, dict(_MOCK_INDICES))
