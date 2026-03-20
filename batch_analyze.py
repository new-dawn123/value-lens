"""Batch-analyze all S&P 500 stocks using the ValueLens pipeline.

Fetches the S&P 500 constituent list from the finviz screener, then runs
each ticker through the full scoring/valuation pipeline.  Results are saved
incrementally to a CSV so the run can be resumed after interruption or
rate-limit stops.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates, check_post_valuation_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import calculate_valuation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds: 2, 4, 8
_FINVIZ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
_FINVIZ_TIMEOUT = 20

_SCREENER_URL = "https://finviz.com/screener.ashx"
_SCREENER_PARAMS = {"v": "111", "f": "idx_sp500", "o": "-marketcap"}
_CONSECUTIVE_FAIL_LIMIT = 5

CSV_COLUMNS = [
    "#",
    "Ticker",
    "Name",
    "Market Cap",
    "Current Price",
    "Fair Value",
    "% vs Fair Value",
    "Fair Price",
    "% vs Fair Price",
    "Score",
]


# ---------------------------------------------------------------------------
# S&P 500 list retrieval via finviz screener
# ---------------------------------------------------------------------------

def _parse_market_cap(text: str) -> float:
    """Convert finviz market cap string (e.g. '3.45T', '120.5B') to a float."""
    text = text.strip()
    if not text or text == "-":
        return 0.0
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    suffix = text[-1].upper()
    if suffix in multipliers:
        try:
            return float(text[:-1]) * multipliers[suffix]
        except ValueError:
            return 0.0
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return 0.0


def _fetch_screener_page(start: int = 1) -> str | None:
    """Fetch a single page of finviz screener results with retry."""
    params = {**_SCREENER_PARAMS, "r": str(start)}
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                _SCREENER_URL,
                params=params,
                headers=_FINVIZ_HEADERS,
                timeout=_FINVIZ_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.text
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** (attempt + 1))
    return None


def fetch_sp500_tickers() -> list[dict]:
    """Scrape the finviz screener for all S&P 500 tickers sorted by market cap.

    Returns a list of dicts with keys: ticker, name, sector, market_cap.
    """
    tickers: list[dict] = []
    start = 1

    while True:
        html = _fetch_screener_page(start)
        if html is None:
            break

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="screener_table")
        if table is None:
            # Try alternative class name
            table = soup.find("table", {"id": "screener-content"})
        if table is None:
            # Fallback: find table by looking at rows with ticker-like content
            tables = soup.find_all("table")
            for t in tables:
                rows = t.find_all("tr")
                if len(rows) > 2:
                    cells = rows[1].find_all("td")
                    if len(cells) >= 10:
                        table = t
                        break

        if table is None:
            break

        rows = table.find_all("tr")[1:]  # skip header
        if not rows:
            break

        page_count = 0
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 10:
                continue
            ticker = cells[1].text.strip()
            name = cells[2].text.strip()
            sector = cells[3].text.strip()
            market_cap_str = cells[6].text.strip()
            if not ticker:
                continue
            tickers.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "market_cap": _parse_market_cap(market_cap_str),
            })
            page_count += 1

        if page_count < 20:
            break  # last page

        start += 20
        time.sleep(1)  # polite delay between screener pages

    # Sort by market cap descending (should already be sorted, but ensure)
    tickers.sort(key=lambda x: x["market_cap"], reverse=True)
    return tickers


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_existing_csv(path: str) -> set[str]:
    """Read already-processed tickers from an existing CSV."""
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            # Skip leading comment lines (e.g. "# Last run: ...") but keep
            # the header row whose first column is named "#".
            while True:
                pos = f.tell()
                line = f.readline()
                if not line or not line.startswith("# "):
                    f.seek(pos)
                    break
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("Ticker", "").strip()
                if ticker:
                    done.add(ticker)
    except Exception:
        pass
    return done


def _write_csv_header(path: str) -> None:
    """Write the CSV header row (with a timestamp comment on line 1)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)


def _append_csv_row(path: str, row: list) -> None:
    """Append a single row to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _count_csv_rows(path: str) -> int:
    """Count data rows in an existing CSV (for resume # numbering)."""
    if not os.path.exists(path):
        return 0
    count = 0
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    count += 1
        return max(count - 1, 0)  # subtract header row
    except Exception:
        return 0


def _format_market_cap(value) -> str:
    """Format market cap number to human-readable string."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return ""
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# Single-ticker analysis
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str) -> list | None:
    """Run the full ValueLens pipeline on a single ticker.

    Returns a list matching CSV_COLUMNS (without the '#' position, which is
    filled in by the main loop), or None if the ticker should be skipped
    (gate failure, negative prices, etc.).
    """
    data = fetch_stock_data(ticker)

    passed, gate_messages = check_gates(data)
    if not passed:
        return None

    scores = score_stock(data)
    valuation = calculate_valuation(data, scores=scores)

    post_passed, post_messages = check_post_valuation_gates(valuation)
    if not post_passed:
        return None

    scores = apply_price_cap(scores, data, valuation)

    current_price = data.get("current_price")
    fair_value = valuation.get("peg_method", {}).get("fair_value")
    fair_price = valuation.get("fair_price")

    pct_vs_fv = ""
    if current_price and fair_value:
        pct_vs_fv = round((current_price - fair_value) / fair_value * 100, 1)

    pct_vs_fp = ""
    if current_price and fair_price:
        pct_vs_fp = round((current_price - fair_price) / fair_price * 100, 1)

    return [
        ticker,
        data.get("name", ""),
        _format_market_cap(data.get("market_cap")),
        current_price or "",
        round(fair_value, 2) if fair_value else "",
        pct_vs_fv,
        round(fair_price, 2) if fair_price else "",
        pct_vs_fp,
        scores.get("final_score", ""),
    ]


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def _process_ticker(ticker: str, order_key: int) -> tuple[int, str, list | None, str]:
    """Worker function: analyze a single ticker with retries.

    Returns (order_key, ticker, row_or_None, status_str).
    row does NOT include the '#' column — that is assigned at flush time.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            row = analyze_ticker(ticker)
            if row is None:
                return order_key, ticker, None, "SKIPPED"
            return order_key, ticker, row, "OK"
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** (attempt + 1))
            last_error = str(e)
    return order_key, ticker, None, f"FAILED: {last_error[:100]}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch-analyze S&P 500 stocks with ValueLens"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the top N tickers by market cap (default: all)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait between submitting tickers (default: 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default="sp500_valuation.csv",
        help="Output CSV file path (default: sp500_valuation.csv)"
    )
    args = parser.parse_args()

    # --- Fetch S&P 500 list ---
    print("Fetching S&P 500 constituent list from finviz...")
    sp500 = fetch_sp500_tickers()
    if not sp500:
        print("Error: could not retrieve S&P 500 list from finviz.")
        sys.exit(1)
    print(f"  Found {len(sp500)} tickers.")

    # Apply limit
    if args.limit:
        sp500 = sp500[: args.limit]
        print(f"  Limited to top {args.limit} by market cap.")

    # --- Resume support ---
    already_done = _read_existing_csv(args.output)
    if already_done:
        remaining = [t for t in sp500 if t["ticker"] not in already_done]
        print(f"  Resuming: {len(already_done)} already processed, "
              f"{len(remaining)} remaining.")
    else:
        remaining = sp500
        _write_csv_header(args.output)

    if not remaining:
        print("All tickers already processed. Nothing to do.")
        sys.exit(0)

    # Ordered list of indices for flush tracking (one per remaining ticker)
    remaining_keys = list(range(len(remaining)))

    total = len(sp500)
    done_count = len(already_done)
    consecutive_failures = 0
    print_lock = threading.Lock()
    stop_event = threading.Event()

    # Buffer for ordered writes: order_key → row (None = skipped/failed)
    pending: dict[int, list | None] = {}
    next_write_idx = 0  # index into remaining_keys for next flush
    csv_row_num = _count_csv_rows(args.output)  # rows already written

    def _flush_pending():
        """Write all consecutive ready results to CSV in order, assigning #."""
        nonlocal next_write_idx, csv_row_num
        while next_write_idx < len(remaining_keys):
            key = remaining_keys[next_write_idx]
            if key not in pending:
                break
            row = pending.pop(key)
            if row is not None:
                csv_row_num += 1
                row.insert(0, csv_row_num)
                _append_csv_row(args.output, row)
            next_write_idx += 1

    print(f"  Processing {len(remaining)} tickers with {args.workers} workers...\n")

    # --- Parallel process loop ---
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, entry in enumerate(remaining):
            if stop_event.is_set():
                break
            ticker = entry["ticker"]
            future = executor.submit(_process_ticker, ticker, idx)
            futures[future] = ticker
            time.sleep(args.delay)  # stagger submissions to avoid burst

        for future in as_completed(futures):
            if stop_event.is_set():
                break

            order_key, ticker, row, status = future.result()
            done_count += 1

            with print_lock:
                print(f"[{done_count}/{total}] {ticker}: {status}")

            # Buffer result and flush any consecutive ready rows
            pending[order_key] = row
            _flush_pending()

            if status.startswith("FAILED"):
                consecutive_failures += 1
                if consecutive_failures >= _CONSECUTIVE_FAIL_LIMIT:
                    with print_lock:
                        print(f"\n{_CONSECUTIVE_FAIL_LIMIT} consecutive failures — "
                              f"likely rate-limited. Stopping.")
                        print(f"Progress saved to {args.output}. "
                              f"Re-run to resume from where you left off.")
                    stop_event.set()
                    # Cancel pending futures
                    for f in futures:
                        f.cancel()
                    sys.exit(2)
            else:
                consecutive_failures = 0

    # Flush any remaining buffered results
    _flush_pending()

    print(f"\nDone! Results saved to {args.output}")


if __name__ == "__main__":
    main()
