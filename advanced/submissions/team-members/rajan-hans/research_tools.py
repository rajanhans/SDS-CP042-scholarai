"""research_tool.py

Provides a simple Yahoo Finance JSON price fetcher.

- Uses requests for the public Yahoo JSON endpoint:
  https://query1.finance.yahoo.com/v7/finance/quote?symbols=SYMBOL

- Exposes a synchronous helper `fetch_yahoo_quote(symbol)` and an async wrapper
  class `YahooPriceTool` with `async def run(input_text)` so it can be used
  directly from asyncio code via `await price_tool.run("TSLA")`.

Note: this module depends on `requests`. Install with:
    python -m pip install requests

"""

from __future__ import annotations

import requests
import asyncio
import re
import time
import random
from typing import Optional, Dict, Any

YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
# Session + headers to appear like a browser and reduce automated blocking
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    }
)

_QUOTE_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 60.0  # seconds (increase TTL to reduce request frequency)


def fetch_yahoo_quote(symbol: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Synchronously fetch a quote for SYMBOL using Yahoo's public JSON endpoint.

    Returns a dict with keys: symbol, price, currency, raw (full quote dict) or
    None on failure.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None

    # Simple in-memory cache to avoid repeated requests in a short time window
    now = time.time()
    cache_entry = _QUOTE_CACHE.get(symbol)
    if cache_entry:
        ts, val = cache_entry
        if now - ts < _CACHE_TTL:
            print(f"[research_tool] cache hit for {symbol}")
            return val

    # small jitter to avoid exact simultaneous bursts
    time.sleep(random.uniform(0.05, 0.25))

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"[research_tool] fetching quote for {symbol} (attempt {attempt}) from {YAHOO_QUOTE_URL}"
            )
            resp = _SESSION.get(
                YAHOO_QUOTE_URL, params={"symbols": symbol}, timeout=timeout
            )
            print(f"[research_tool] response status: {resp.status_code}")

            # Handle rate limit explicitly
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = min(30.0, 2**attempt + random.random() * 3)
                else:
                    # longer wait when no Retry-After provided
                    wait = min(30.0, 2**attempt + random.random() * 3)
                print(
                    f"[research_tool] rate limited (429). Waiting for {wait:.1f}s before retrying"
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            # Debug: show if we got expected keys
            print(f"[research_tool] got json keys: {list(data.keys())}")
            result = data.get("quoteResponse", {}).get("result", [])
            if not result:
                print(f"[research_tool] no result for symbol {symbol}")
                return None
            quote = result[0]
            price = quote.get("regularMarketPrice")
            currency = quote.get("currency")
            out = {"symbol": symbol, "price": price, "currency": currency, "raw": quote}
            # Cache the result
            _QUOTE_CACHE[symbol] = (time.time(), out)
            print(f"[research_tool] returning: {out}")
            return out
        except requests.HTTPError as e:
            # If it's a 5xx, retry a few times; otherwise break
            status = getattr(e.response, "status_code", None)
            print(f"[research_tool] HTTPError for {symbol}: {e} (status={status})")
            if status and 500 <= status < 600 and attempt < max_attempts:
                backoff = min(30.0, 2**attempt + random.random() * 3)
                print(f"[research_tool] server error, retrying after {backoff:.1f}s")
                time.sleep(backoff)
                continue
            return None
        except Exception as e:
            print(f"[research_tool] exception while fetching quote for {symbol}: {e}")
            if attempt < max_attempts:
                backoff = min(30.0, 2**attempt + random.random() * 3)
                time.sleep(backoff)
                continue
            return None


class YahooPriceTool:
    """Async-friendly wrapper around fetch_yahoo_quote.

    Usage:
        tool = YahooPriceTool()
        info = await tool.run("TSLA")
    """

    async def run(self, input_text: str) -> Optional[Dict[str, Any]]:
        # Accept either a symbol string or free text; extract first token as symbol
        symbol = (input_text or "").strip().split()[0].upper() if input_text else ""
        if not symbol:
            return None
        # Run the blocking requests call in a thread
        print(f"[research_tool] running price tool for symbol: {symbol}")
        res = await asyncio.to_thread(fetch_yahoo_quote, symbol)
        print(f"[research_tool] async run got (yahoo): {res}")
        if res is not None:
            return res

        # Fallback: try Robinhood scraping
        print(
            f"[research_tool] yahoo returned no data; attempting Robinhood fallback for {symbol}"
        )
        rh = await asyncio.to_thread(fetch_robinhood_price, symbol)
        print(f"[research_tool] async run got (robinhood): {rh}")
        return rh


def fetch_robinhood_price(
    symbol: str, timeout: float = 5.0
) -> Optional[Dict[str, Any]]:
    """Fetch price from Robinhood stock page by scraping the HTML.

    This is a best-effort fallback. Robinhood pages are often client-rendered
    and may block automated clients. We try several JSON/regex patterns to find
    a numeric price in the returned HTML.
    """
    url = f"https://robinhood.com/us/en/stocks/{symbol}"
    try:
        print(f"[research_tool] fetching Robinhood page for {symbol}: {url}")
        resp = _SESSION.get(url, timeout=timeout)
        print(f"[research_tool] robinhood response status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[research_tool] robinhood non-200 status: {resp.status_code}")
            return None

        text = resp.text

        # Try several regex patterns commonly found in embedded JSON data
        patterns = [
            r'"last_trade_price"\s*:\s*"?([0-9]+\.[0-9]+)"?',
            r'"last_trade_price"\s*:\s*([0-9]+\.[0-9]+)',
            r'"intraday_latest_price"\s*:\s*"?([0-9]+\.[0-9]+)"?',
            r'"last_price"\s*:\s*"?([0-9]+\.[0-9]+)"?',
            r'"price"\s*:\s*"?([0-9]+\.[0-9]+)"?',
        ]

        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    price = float(m.group(1).replace(",", ""))
                    out = {
                        "symbol": symbol,
                        "price": price,
                        "currency": "USD",
                        "raw": None,
                    }
                    print(
                        f"[research_tool] robinhood parsed price via regex '{pat}': {price}"
                    )
                    return out
                except Exception:
                    continue

        # As a last resort, try to find a visible price-like pattern in the HTML
        m = re.search(r">([0-9]{1,5}\.[0-9]{2})<", text)
        if m:
            try:
                price = float(m.group(1).replace(",", ""))
                out = {"symbol": symbol, "price": price, "currency": "USD", "raw": None}
                print(
                    f"[research_tool] robinhood parsed fallback visible price: {price}"
                )
                return out
            except Exception:
                pass

        print(f"[research_tool] robinhood: failed to extract price for {symbol}")
        return None
    except Exception as e:
        print(f"[research_tool] robinhood exception for {symbol}: {e}")
        return None
