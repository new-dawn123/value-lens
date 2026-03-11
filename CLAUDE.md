# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ValueLens is a stock fundamentals analyzer using a PEG-based scoring model. It has two interfaces: a Streamlit web UI (`app.py`) and a CLI tool (`analyze.py`).

## Commands

```bash
# Setup
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
pip install -r requirements.txt

# Run web UI (port 8501)
streamlit run app.py

# Run CLI
python analyze.py AAPL
python analyze.py AAPL --eps 5.00 --growth 15.0
python analyze.py AAPL --no-hist-premium

# Windows launchers (auto-install deps, use local venv)
launcher.bat        # Interactive CLI loop
launcher-gui.bat    # Streamlit GUI
```

No tests, linter, or build step exist.

## Architecture

**Pipeline:** `Fetch → Gates → Score → Valuate → Price Cap → Output`

Both entry points (`app.py`, `analyze.py`) follow the same pipeline through five core modules in `src/`:

| Module | Responsibility |
|---|---|
| `data_fetcher.py` | Fetches from yfinance (primary) + scrapes finviz.com (growth metrics fallback). Handles FX conversion for ADRs. Retry with exponential backoff for finviz. |
| `gates.py` | Hard gates (block scoring: EPS <= 0, no growth estimate) and soft gates (warnings: FX conversion, currency mismatch, penny stock). |
| `scorer.py` | Scores 0-100 from three weighted components: PEG (70%), EPS Revisions (20%), Earnings Surprises (10%). Contains the growth dampening function and `_fair_pe()` compounding model. |
| `valuator.py` | Fair Value = PEG fair price x historical premium. Entry price from beta-scaled growth scenario. Exit price from historical P/E stretch. |
| `formatter.py` | Rich terminal output for CLI. Not used by Streamlit (app.py renders its own UI with Plotly charts). |

**Key constants** in `scorer.py`: `BASE_PE = 12.0` (zero-growth P/E anchor), `_GROWTH_DAMPEN_THRESHOLD = 15.0` (no dampening up to 15%), `_GROWTH_DAMPEN_K = 30.0` (compression factor for excess above threshold). Historical premium in `valuator.py` clamped to ±20%.

**Data flow:** All modules communicate via plain `dict` objects — `data` (fetched metrics), `scores` (scoring output), `valuation` (pricing output).

## Streamlit-specific Patterns

- `@st.cache_data(ttl=3600)` caches stock fetches for 1 hour
- Cache version is incremented per-ticker on finviz failure to force retry on next submit
- Session state holds rendered results so they survive Streamlit reruns
- HTML rendered via `st.markdown(..., unsafe_allow_html=True)` for styled tables and headers

## Technical Documentation

Detailed algorithm docs live in `docs/`:
- `algorithms.md` — PEG model, growth dampening, gate system
- `data-sources.md` — Field mapping for yfinance and finviz
- `scoring.md` — Score formula, component weights, score ranges
- `valuation.md` — Fair price model, historical premium, entry/exit logic

## Tech Stack

Python 3.9+, Streamlit, yfinance, pandas, Plotly, Rich, BeautifulSoup4 (finviz scraping).
