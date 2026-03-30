import base64
import html as html_mod
import statistics
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates, check_post_valuation_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import (
    calculate_valuation,
    compute_historical_forward_pe_series,
    compute_historical_pe_series,
)

from datetime import date, timedelta

from src.warrant_pricer import (
    greeks,
    implied_volatility,
    scenario_price,
)

_PROJECT_DIR = Path(__file__).parent
_LOGO_PATH = _PROJECT_DIR / "assets" / "logo.png"
_CSV_PATH = _PROJECT_DIR / "top500_valuation.csv"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(ticker: str, cache_version: int = 0) -> dict:
    """Cache stock data for 1 hour.

    cache_version is incremented when finviz failed on a previous fetch,
    so the next submit bypasses the stale cache and retries finviz.
    """
    return fetch_stock_data(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_fx_rate(from_ccy: str, to_ccy: str) -> float | None:
    """Fetch FX rate via yfinance. Returns rate such that 1 from_ccy = rate to_ccy."""
    if from_ccy == to_ccy:
        return 1.0
    import yfinance as yf
    pair = f"{from_ccy}{to_ccy}=X"
    try:
        hist = yf.Ticker(pair).history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    # Try inverse
    pair_inv = f"{to_ccy}{from_ccy}=X"
    try:
        hist = yf.Ticker(pair_inv).history(period="1d")
        if not hist.empty:
            return 1.0 / float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Batch display helpers
# ---------------------------------------------------------------------------

def _pct_to_bg(val: float) -> str:
    """Map a percentage to a green-yellow-red background.

    Green at -10% or below, yellow at +15%, red at +40% or above.
    """
    if pd.isna(val):
        return ""
    MIN_V, MAX_V = -10.0, 40.0
    MID_V = (MIN_V + MAX_V) / 2.0  # 15.0
    clamped = max(MIN_V, min(MAX_V, float(val)))

    # Interpolate color: green(76,175,80) → yellow(255,235,59) → red(244,67,54)
    if clamped <= MID_V:
        t = (clamped - MIN_V) / (MID_V - MIN_V)  # 0..1
        r = int(76 + t * (255 - 76))
        g = int(175 + t * (235 - 175))
        b = int(80 + t * (59 - 80))
    else:
        t = (clamped - MID_V) / (MAX_V - MID_V)  # 0..1
        r = int(255 + t * (244 - 255))
        g = int(235 + t * (67 - 235))
        b = int(59 + t * (54 - 59))

    return f"color: rgb({r},{g},{b})"


def _apply_valuation_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Return a same-shaped DataFrame of CSS styles with green/yellow/red gradients."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for pct_col, price_col in [
        ("% vs Fair Value", "Fair Value"),
        ("% vs Fair Price", "Fair Price"),
    ]:
        if pct_col in df.columns:
            styles[pct_col] = df[pct_col].map(_pct_to_bg)
        if price_col in df.columns and pct_col in df.columns:
            styles[price_col] = df[pct_col].map(_pct_to_bg)
    return styles


def _read_batch_run_date() -> str | None:
    """Read the 'Last run' timestamp from the CSV comment header."""
    if not _CSV_PATH.exists():
        return None
    try:
        with open(_CSV_PATH, "r", encoding="utf-8") as f:
            first = f.readline()
        if first.startswith("# Last run:"):
            return first.removeprefix("# Last run:").strip()
    except Exception:
        pass
    return None


def _load_batch_csv() -> pd.DataFrame | None:
    """Load and prepare the batch CSV. Returns None if unavailable."""
    if not _CSV_PATH.exists():
        return None

    try:
        # Skip leading comment lines (e.g. "# Last run: ...") but keep the
        # header row which also starts with "#" (the column name).
        skip = 0
        with open(_CSV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    skip += 1
                else:
                    break
        df = pd.read_csv(_CSV_PATH, skiprows=skip, on_bad_lines="skip")
    except Exception:
        return None

    if df.empty:
        return None

    # Coerce numeric columns (empty strings from gate failures become NaN)
    numeric_cols = [
        "Current Price", "Fair Value", "% vs Fair Value",
        "Fair Price", "% vs Fair Price", "Score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --- Page config ---
st.set_page_config(page_title="ValueLens", page_icon=str(_LOGO_PATH), layout="wide")

st.markdown(
    "<style>"
    "section.main > div.block-container,"
    "[data-testid='stMainBlockContainer'],"
    "[data-testid='stAppViewBlockContainer'] {"
    "  padding-left: 15% !important;"
    "  padding-right: 15% !important;"
    "  max-width: 100% !important;"
    "}"
    "</style>",
    unsafe_allow_html=True,
)

# Select all text on focus for number inputs (must use components.html for JS)
import streamlit.components.v1 as _components
_components.html(
    "<script>"
    "const doc = window.parent.document;"
    "doc.addEventListener('focusin', function(e) {"
    "  if (e.target.tagName === 'INPUT'"
    "      && (e.target.type === 'number' || e.target.type === 'text')"
    "      && e.target.closest('[data-testid]')) {"
    "    setTimeout(function() { e.target.select(); }, 10);"
    "  }"
    "});"
    "</script>",
    height=0,
)

_logo_b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
st.markdown(
    f"<link href='https://fonts.googleapis.com/css2?family=Inter:wght@600&display=swap'"
    f" rel='stylesheet'>"
    f"<div style='display:flex;align-items:center;gap:16px;margin-top:-2rem;margin-bottom:4px'>"
    f"<img src='data:image/png;base64,{_logo_b64}'"
    f" style='width:52px;height:52px;object-fit:contain'>"
    f"<span style='font-family:Inter,sans-serif;font-size:2.6em;font-weight:600;"
    f"line-height:1;letter-spacing:-0.02em'>"
    f"<span style='color:#1c6295'>Value</span>"
    f"<span style='color:#50a375'>Lens</span></span>"
    f"</div>",
    unsafe_allow_html=True,
)

# --- Tabs ---
tab_analyzer, tab_batch, tab_warrant = st.tabs(["Stock Analysis", "Screener", "Warrant Analysis"])

# ===== Analyzer tab =====
with tab_analyzer:

    # --- Handle deep-link from batch table ---
    _qp_ticker = st.query_params.get("ticker", "")

    # Seed the text input from the query-param only once so that
    # subsequent reruns never overwrite what the user typed.
    if _qp_ticker and "_ticker_seeded" not in st.session_state:
        st.session_state["_ticker_input"] = _qp_ticker
        st.session_state["_ticker_seeded"] = True

    # --- Section 1: Input bar ---
    with st.form("analyze_form"):
        col_ticker, col_button = st.columns([3, 1])
        with col_ticker:
            ticker = st.text_input(
                "Ticker", key="_ticker_input", placeholder="e.g. NVDA",
                label_visibility="collapsed",
            )
        with col_button:
            analyze = st.form_submit_button("Analyze", type="primary", width="stretch")

    with st.expander("Advanced Options"):
        # Pull actual values from last result for defaults
        _prev = st.session_state.get("result", {}).get("data", {})
        _default_eps = round(_prev["trailing_eps"], 2) if _prev.get("trailing_eps") else 1.00
        _default_growth = round(_prev.get("growth_5y") or 15.0, 1)

        use_custom_eps = st.checkbox(
            "Override trailing EPS",
            help="Override trailing EPS for stocks with GAAP distortions.",
        )
        custom_eps = None
        if use_custom_eps:
            custom_eps = st.number_input(
                "Custom EPS",
                min_value=0.01,
                value=_default_eps,
                step=0.01,
                format="%.2f",
                placeholder="e.g. 5.67",
            )
        use_custom_growth = st.checkbox(
            "Override 5Y growth rate",
            help="Bypasses dampened 5Y growth calculation.",
        )
        custom_growth = None
        if use_custom_growth:
            custom_growth = st.number_input(
                "Custom 5Y Growth (%)",
                min_value=0.1,
                value=_default_growth,
                step=0.5,
                format="%.1f",
                placeholder="e.g. 15.0",
            )
        disregard_hist_premium = st.checkbox(
            "Disregard Historical Premium",
            help="Still calculated and shown, but not applied to Fair Price.",
        )
        uncap_hist_premium = st.checkbox(
            "Uncap Historical Premium",
            help="Remove the \u00b120% clamp (0.80\u20131.20) on the historical premium.",
        )

    # Auto-analyze: deep-link first visit, or re-run when options change
    _auto_done = st.session_state.get("_auto_analyzed_ticker")
    _opts_fingerprint = (use_custom_eps, custom_eps, use_custom_growth,
                         custom_growth, disregard_hist_premium, uncap_hist_premium)
    _prev_fingerprint = st.session_state.get("_opts_fingerprint")
    if _qp_ticker and not analyze:
        _is_deep_link = _auto_done != _qp_ticker
        _opts_changed = ("result" in st.session_state
                         and _opts_fingerprint != _prev_fingerprint)
        if _is_deep_link or _opts_changed:
            analyze = True
            ticker = _qp_ticker
        if _is_deep_link:
            st.session_state["_auto_analyzed_ticker"] = _qp_ticker

    # --- Orchestration ---
    # NOTE: Do not use st.stop() — it kills the entire script and
    # prevents the Screener tab from rendering.
    if analyze and ticker:
        ticker = ticker.upper().strip()
        st.query_params["ticker"] = ticker
        _abort = False

        fetch_error = None
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                # Bust cache for tickers where finviz previously failed
                finviz_fails = st.session_state.setdefault("_finviz_fails", {})
                cache_ver = finviz_fails.get(ticker, 0)
                data = cached_fetch(ticker, cache_version=cache_ver)

                if not data.get("_finviz_ok", True):
                    # Next submit will use a new cache key -> retry finviz
                    finviz_fails[ticker] = cache_ver + 1
                else:
                    finviz_fails.pop(ticker, None)
            except Exception as e:
                fetch_error = e

        if fetch_error is not None:
            st.error(f"Error fetching data for {ticker}: {fetch_error}")
            _abort = True

        if not _abort:
            passed, gate_messages = check_gates(data, custom_growth=custom_growth)
            if not passed:
                st.error(f"**{ticker} \u2014 {data.get('name', 'Unknown')}**")
                for msg in gate_messages:
                    st.error(msg)
                _abort = True

        if not _abort:
            scores = score_stock(data, custom_eps=custom_eps, custom_growth=custom_growth)
            valuation = calculate_valuation(
                data, custom_eps=custom_eps, custom_growth=custom_growth, scores=scores,
                disregard_hist_premium=disregard_hist_premium,
                uncap_hist_premium=uncap_hist_premium,
            )
            post_passed, post_messages = check_post_valuation_gates(valuation)
            if not post_passed:
                st.error(f"**{ticker} — {data.get('name', 'Unknown')}**")
                for msg in post_messages:
                    st.error(msg)
                _abort = True

        if _abort:
            st.session_state.pop("result", None)
        else:
            scores = apply_price_cap(scores, data, valuation)
            pe_series = compute_historical_pe_series(data)
            fwd_pe_series = compute_historical_forward_pe_series(data)

            st.session_state["result"] = {
                "ticker": ticker,
                "data": data,
                "scores": scores,
                "valuation": valuation,
                "pe_series": pe_series,
                "fwd_pe_series": fwd_pe_series,
                "gate_messages": gate_messages,
                "disregard_hist_premium": disregard_hist_premium,
            }
            st.session_state["_opts_fingerprint"] = _opts_fingerprint

    # --- Rendering (from session state) ---
    if "result" in st.session_state:
        try:
            r = st.session_state["result"]
            ticker = r["ticker"]
            data = r["data"]
            scores = r["scores"]
            valuation = r["valuation"]
            pe_series = r["pe_series"]
            fwd_pe_series = r["fwd_pe_series"]
            disregard_hist_premium = r.get("disregard_hist_premium", False)

            # --- Section 2: Score & Summary ---
            name = data.get("name", ticker)
            price = data.get("current_price")

            website = data.get("website")
            logo_url = ""
            if website:
                domain = urlparse(website).netloc or urlparse(website).path
                logo_url = f"https://www.google.com/s2/favicons?domain={html_mod.escape(domain, quote=True)}&sz=128"

            if logo_url:
                _esc_ticker = html_mod.escape(ticker)
                _esc_name = html_mod.escape(name)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px'>"
                    f"<img src='{logo_url}' width='40' height='40' style='border-radius:6px'"
                    f" onerror=\"this.style.display='none'\">"
                    f"<span style='font-size:1.5em;font-weight:600'>{_esc_ticker} \u2014 {_esc_name}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"### {ticker} \u2014 {name}")
            score = scores["final_score"]
            label = scores["label"]

            score_colors = {
                "Strong Buy": "green", "Attractive": "green",
                "Hold": "orange", "Unattractive": "red", "Avoid": "red",
            }
            score_color = score_colors.get(label, "gray")
            _esc_label = html_mod.escape(label)
            st.markdown(
                f"<h2 style='color:{score_color};margin-top:0'>{score}/100 \u2014 {_esc_label}</h2>",
                unsafe_allow_html=True,
            )

            # Metric cards — HTML flexbox for full-width space-between
            peg_m_top = valuation.get("peg_method", {})
            hist_p_top = valuation.get("historical_premium", {})
            net_cash_top = valuation.get("net_cash_per_share")

            def _metric_card(label_txt, value_txt, deltas=None):
                """Return HTML for a single metric card matching st.metric style."""
                _l = html_mod.escape(str(label_txt))
                _v = html_mod.escape(str(value_txt))
                card = (
                    f"<div>"
                    f"<div style='font-size:0.9rem;color:var(--text-color, rgba(224,228,238,0.6));margin-bottom:4px'>{_l}</div>"
                    f"<div style='font-size:2.25rem;font-weight:600;line-height:1.2'>{_v}</div>"
                )
                for d in (deltas or []):
                    arrow = "" if d.get("no_arrow") else ("&#x2193; " if d.get("down") else "&#x2191; ")
                    _dt = html_mod.escape(str(d['text']))
                    card += (
                        f"<div style='font-size:0.85rem;color:var(--text-color, rgba(224,228,238,0.7));"
                        f"background:rgba(255,255,255,0.08);border-radius:16px;"
                        f"padding:2px 8px;margin-top:4px;display:inline-block'>"
                        f"{arrow}{_dt}</div><br>"
                    )
                card += "</div>"
                return card

            cp_val = f"${price:,.2f}" if price else "N/A"
            fv_raw = peg_m_top.get("fair_value")
            fv_val = f"${fv_raw:,.2f}" if fv_raw else "N/A"

            fp = valuation.get("fair_price")
            fp_val = f"${fp:,.2f}" if fp else "N/A"
            fp_deltas = []
            premium_top = hist_p_top.get("premium")
            raw_premium_top = hist_p_top.get("raw_premium")
            if premium_top is not None:
                is_capped = raw_premium_top is not None and round(raw_premium_top, 2) != round(premium_top, 2)
                prem_pct = abs(round((premium_top - 1) * 100))
                txt = f"{prem_pct}%"
                if is_capped:
                    raw_pct = abs(round((raw_premium_top - 1) * 100))
                    txt += f" ({raw_pct}%)"
                txt += " premium"
                if not disregard_hist_premium:
                    fp_deltas.append({"text": txt, "down": premium_top < 1})
            if net_cash_top is not None:
                fp_deltas.append({"text": f"${abs(net_cash_top):,.2f} net cash", "down": net_cash_top < 0})

            ep = valuation.get("entry_price")
            margin_pct = round(valuation.get("margin_of_safety", 0) * 100)
            ep_val = f"≤ ${ep:,.2f}" if ep else "N/A"

            xp = valuation.get("exit_price")
            exit_pct = round(valuation.get("exit_premium", 0) * 100)
            xp_val = f"≥ ${xp:,.2f}" if xp else "N/A"

            peg_val = scores.get("peg")
            cp_deltas = [{"text": f"PEG {peg_val:.2f}", "no_arrow": True}] if peg_val is not None else []
            eff_g = scores.get("blended_growth")
            fv_deltas = [{"text": f"Growth {eff_g:.1f}%", "no_arrow": True}] if eff_g is not None else []

            cards = (
                _metric_card("Current Price", cp_val, cp_deltas)
                + _metric_card("Fair Value", fv_val, fv_deltas)
                + _metric_card("Fair Price", fp_val, fp_deltas)
                + _metric_card("Entry Price", ep_val, [{"text": f"{margin_pct}% margin", "down": True}])
                + _metric_card("Exit Price", xp_val, [{"text": f"{exit_pct}% above entry", "down": False}])
            )
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;gap:1rem'>{cards}</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            # --- Section 3: Price chart ---
            historical_prices = data.get("historical_prices")
            if historical_prices:
                dates = [p["date"] for p in historical_prices]
                prices = [p["price"] for p in historical_prices]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=prices, mode="lines",
                    name="Price", line=dict(color="#42A5F5", width=2),
                ))

                fv = valuation.get("fair_price")
                ep = valuation.get("entry_price")
                xp = valuation.get("exit_price")

                if ep:
                    fig.add_hline(
                        y=ep, line_dash="dash", line_color="#4DB6AC",
                        annotation_text=f"Entry ${ep:,.0f}",
                        annotation_position="bottom right",
                    )
                if fv:
                    fig.add_hline(
                        y=fv, line_dash="solid", line_color="#FFB74D",
                        annotation_text=f"Fair Price ${fv:,.0f}",
                        annotation_position="top right",
                    )
                if xp:
                    fig.add_hline(
                        y=xp, line_dash="dash", line_color="#EF5350",
                        annotation_text=f"Exit ${xp:,.0f}",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    title=f"{ticker} \u2014 Price vs Fair Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    showlegend=False,
                )

                st.plotly_chart(fig, width="stretch")

            # --- Section 4: Historical P/E charts (side by side) ---
            pe_col_left, pe_col_right = st.columns(2)

            with pe_col_left:
                if pe_series and len(pe_series) >= 6:
                    pe_dates = [p["date"] for p in pe_series]
                    pe_values = [p["pe"] for p in pe_series]
                    median_pe = statistics.median(pe_values)

                    fig_pe = go.Figure()
                    fig_pe.add_trace(go.Scatter(
                        x=pe_dates, y=pe_values, mode="lines",
                        name="P/E", line=dict(color="#9C27B0", width=2),
                    ))
                    current_pe = data.get("trailing_pe")
                    median_pos = "top right" if not current_pe or median_pe >= current_pe else "bottom right"
                    fig_pe.add_hline(
                        y=median_pe, line_dash="dash", line_color="gray",
                        annotation_text=f"Median P/E {median_pe:.1f}",
                        annotation_position=median_pos,
                    )

                    if current_pe:
                        current_pos = "top right" if current_pe >= median_pe else "bottom right"
                        fig_pe.add_hline(
                            y=current_pe, line_dash="dot", line_color="orange",
                            annotation_text=f"Current P/E {current_pe:.1f}",
                            annotation_position=current_pos,
                        )

                    fig_pe.update_layout(
                        title=f"{ticker} \u2014 Historical P/E",
                        xaxis_title="Date",
                        yaxis_title="P/E Ratio",
                        height=350,
                        showlegend=False,
                    )

                    st.plotly_chart(fig_pe, width="stretch")
                else:
                    st.info("Historical P/E chart unavailable \u2014 insufficient positive earnings history.")

            with pe_col_right:
                if fwd_pe_series and len(fwd_pe_series) >= 6:
                    fwd_dates = [p["date"] for p in fwd_pe_series]
                    fwd_values = [p["pe"] for p in fwd_pe_series]
                    fwd_median = statistics.median(fwd_values)

                    fig_fwd = go.Figure()
                    fig_fwd.add_trace(go.Scatter(
                        x=fwd_dates, y=fwd_values, mode="lines",
                        name="Forward P/E", line=dict(color="#00897B", width=2),
                    ))
                    current_fwd_pe = data.get("forward_pe")
                    fwd_median_pos = "top right" if not current_fwd_pe or fwd_median >= current_fwd_pe else "bottom right"
                    fig_fwd.add_hline(
                        y=fwd_median, line_dash="dash", line_color="gray",
                        annotation_text=f"Median Fwd P/E {fwd_median:.1f}",
                        annotation_position=fwd_median_pos,
                    )

                    if current_fwd_pe:
                        fwd_current_pos = "top right" if current_fwd_pe >= fwd_median else "bottom right"
                        fig_fwd.add_hline(
                            y=current_fwd_pe, line_dash="dot", line_color="orange",
                            annotation_text=f"Current Fwd P/E {current_fwd_pe:.1f}",
                            annotation_position=fwd_current_pos,
                        )

                    fig_fwd.update_layout(
                        title=f"{ticker} \u2014 Historical Forward P/E",
                        xaxis_title="Date",
                        yaxis_title="Forward P/E",
                        height=350,
                        showlegend=False,
                    )

                    st.plotly_chart(fig_fwd, width="stretch")
                else:
                    st.info("Historical Forward P/E chart unavailable \u2014 insufficient earnings data.")

            # --- Section 4b: Annual Revenue & EPS histograms (side by side) ---
            rev_col, eps_col = st.columns(2)

            with rev_col:
                rev_hist = data.get("annual_revenue_history")
                if rev_hist and len(rev_hist) >= 2:
                    rev_years = [r["date"].strftime("%FY%y") if hasattr(r["date"], "strftime") else str(r["date"]) for r in rev_hist]
                    rev_labels = [r["date"].strftime("FY%y") for r in rev_hist]
                    rev_values = [r["revenue"] for r in rev_hist]
                    # YOY growth
                    rev_yoy = [None] + [
                        ((rev_values[i] - rev_values[i - 1]) / abs(rev_values[i - 1])) * 100
                        if rev_values[i - 1] != 0 else None
                        for i in range(1, len(rev_values))
                    ]

                    from plotly.subplots import make_subplots
                    rev_billions = [v / 1e9 for v in rev_values]
                    fig_rev = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_rev.add_trace(go.Bar(
                        x=rev_labels, y=rev_billions,
                        name="Revenue", marker_color="#00796B",
                        hovertemplate="%{x}: $%{y:.1f}B<extra></extra>",
                    ), secondary_y=False)
                    # YOY growth line (skip first None)
                    growth_labels = [rev_labels[i] for i in range(len(rev_yoy)) if rev_yoy[i] is not None]
                    growth_values = [round(v, 1) for v in rev_yoy if v is not None]
                    fig_rev.add_trace(go.Scatter(
                        x=growth_labels, y=growth_values, mode="lines+markers",
                        name="YOY Growth", line=dict(color="#FFA500", width=2),
                        marker=dict(size=4),
                        hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
                    ), secondary_y=True)
                    # Add percentage labels on growth points
                    for lbl, val in zip(growth_labels, growth_values):
                        fig_rev.add_annotation(
                            x=lbl, y=val, yref="y2",
                            text=f"{val:+.1f}%", showarrow=False,
                            yshift=16, font=dict(color="#FFA500"),
                        )
                    rev_max = max(rev_billions)
                    # Center growth line in the middle of the y2 axis
                    g_mid = (max(growth_values) + min(growth_values)) / 2
                    g_half = max((max(growth_values) - min(growth_values)) / 2, abs(g_mid) * 0.5, 5)
                    fig_rev.update_layout(
                        title=f"{ticker} — Annual Revenue",
                        height=380,
                        showlegend=False,
                        yaxis=dict(title="Revenue ($B)", showgrid=False, range=[0, rev_max * 1.22]),
                        yaxis2=dict(title="YOY Growth (%)", showgrid=False,
                                    range=[g_mid - g_half * 2, g_mid + g_half * 2]),
                    )
                    st.plotly_chart(fig_rev, width="stretch")
                else:
                    st.info("Annual revenue chart unavailable — insufficient data.")

            with eps_col:
                eps_hist = data.get("annual_eps_history")
                if eps_hist and len(eps_hist) >= 2:
                    eps_labels = [r["date"].strftime("FY%y") for r in eps_hist]
                    eps_values = [r["eps"] for r in eps_hist]
                    # YOY growth
                    eps_yoy = [None] + [
                        ((eps_values[i] - eps_values[i - 1]) / abs(eps_values[i - 1])) * 100
                        if eps_values[i - 1] != 0 else None
                        for i in range(1, len(eps_values))
                    ]

                    from plotly.subplots import make_subplots
                    fig_eps = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_eps.add_trace(go.Bar(
                        x=eps_labels, y=eps_values,
                        name="EPS", marker_color="#1565C0",
                        hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
                    ), secondary_y=False)
                    # YOY growth line
                    eg_labels = [eps_labels[i] for i in range(len(eps_yoy)) if eps_yoy[i] is not None]
                    eg_values = [round(v, 1) for v in eps_yoy if v is not None]
                    fig_eps.add_trace(go.Scatter(
                        x=eg_labels, y=eg_values, mode="lines+markers",
                        name="YOY Growth", line=dict(color="#FFA500", width=2),
                        marker=dict(size=4),
                        hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
                    ), secondary_y=True)
                    for lbl, val in zip(eg_labels, eg_values):
                        fig_eps.add_annotation(
                            x=lbl, y=val, yref="y2",
                            text=f"{val:+.1f}%", showarrow=False,
                            yshift=16, font=dict(color="#FFA500"),
                        )
                    eps_max = max(eps_values)
                    eps_min = min(eps_values)
                    eps_y_lo = eps_min * 1.22 if eps_min < 0 else 0
                    # Center growth line in the middle of the y2 axis
                    eg_mid = (max(eg_values) + min(eg_values)) / 2
                    eg_half = max((max(eg_values) - min(eg_values)) / 2, abs(eg_mid) * 0.5, 5)
                    fig_eps.update_layout(
                        title=f"{ticker} — Annual EPS",
                        height=380,
                        showlegend=False,
                        yaxis=dict(title="EPS ($)", showgrid=False, range=[eps_y_lo, eps_max * 1.22]),
                        yaxis2=dict(title="YOY Growth (%)", showgrid=False,
                                    range=[eg_mid - eg_half * 2, eg_mid + eg_half * 2]),
                    )
                    st.plotly_chart(fig_eps, width="stretch")
                else:
                    st.info("Annual EPS chart unavailable — insufficient data.")

            # --- Section 5: Detailed Metrics (expandable) ---
            with st.expander("Detailed Metrics"):

                peg_m = valuation.get("peg_method", {})
                hist_p = valuation.get("historical_premium", {})

                def _html_table(title: str, metrics: dict) -> str:
                    """Build an HTML metric table block."""
                    _esc = html_mod.escape
                    rows = "".join(
                        f"<tr><td style='padding:4px 8px'>{_esc(str(k))}</td>"
                        f"<td style='padding:4px 8px;text-align:right'>{_esc(str(v))}</td></tr>"
                        for k, v in metrics.items()
                    )
                    return (
                        f"<div style='break-inside:avoid;margin-bottom:24px'>"
                        f"<span style='font-weight:800;font-size:1.05em'>{_esc(title)}</span>"
                        f"<table style='width:100%;table-layout:fixed;border-collapse:collapse;margin:6px 0 0 0'>"
                        f"<colgroup><col style='width:65%'><col style='width:35%'></colgroup>"
                        f"<tr><th style='padding:4px 8px;text-align:left;font-weight:600'>Metric</th>"
                        f"<th style='padding:4px 8px;text-align:right;font-weight:600'>Value</th></tr>"
                        f"{rows}</table></div>"
                    )

                def _html_breakdown_table(title: str, rows_data: list[dict]) -> str:
                    """Build an HTML scoring breakdown table."""
                    _esc = html_mod.escape
                    header = "<tr><th style='padding:4px 8px;text-align:left;font-weight:600'>Component</th><th style='padding:4px 8px;text-align:right;font-weight:600'>Score</th><th style='padding:4px 8px;text-align:right;font-weight:600'>Weight</th></tr>"
                    rows = "".join(
                        f"<tr><td style='padding:4px 8px'>{_esc(str(r['Component']))}</td>"
                        f"<td style='padding:4px 8px;text-align:right'>{_esc(str(r['Score']))}</td>"
                        f"<td style='padding:4px 8px;text-align:right'>{_esc(str(r['Weight']))}</td></tr>"
                        for r in rows_data
                    )
                    return (
                        f"<div style='break-inside:avoid;margin-bottom:24px'>"
                        f"<span style='font-weight:800;font-size:1.05em'>{_esc(title)}</span>"
                        f"<table style='width:100%;table-layout:fixed;border-collapse:collapse;margin:6px 0 0 0'>"
                        f"<colgroup><col style='width:50%'><col style='width:25%'><col style='width:25%'></colgroup>"
                        f"{header}{rows}</table></div>"
                    )

                tbl_earnings = _html_table("Earnings & Price", {
                    "Trailing EPS": f"${data['trailing_eps']:.2f}" if data.get("trailing_eps") else "N/A",
                    "Trailing P/E": f"{data['trailing_pe']:.2f}" if data.get("trailing_pe") else "N/A",
                    "Forward P/E": f"{data['forward_pe']:.2f}" if data.get("forward_pe") else "N/A",
                    "P/S Ratio": f"{data['ps_ratio']:.2f}" if data.get("ps_ratio") else "N/A",
                    "Beta": f"{data['beta']:.2f}" if data.get("beta") else "N/A",
                })

                tbl_growth = _html_table("Growth", {
                    "0Y EPS Growth": f"{data['growth_current_year']:.1f}%" if data.get("growth_current_year") else "N/A",
                    "1Y EPS Growth": f"{data['growth_next_year']:.1f}%" if data.get("growth_next_year") else "N/A",
                    "5Y EPS Growth": f"{data['growth_5y']:.1f}%" if data.get("growth_5y") else "N/A",
                    "1Y Revenue Growth": f"{data['revenue_growth_next_year']:.1f}%" if data.get("revenue_growth_next_year") else "N/A",
                    "Past 3Y EPS Growth": f"{data['historical_growth_3y']:.1f}%" if data.get("historical_growth_3y") else "N/A",
                    "Past 5Y EPS Growth": f"{data['historical_growth_5y']:.1f}%" if data.get("historical_growth_5y") else "N/A",
                    "Past 3Y Sales Growth": f"{data['sales_growth_3y']:.1f}%" if data.get("sales_growth_3y") else "N/A",
                    "Past 5Y Sales Growth": f"{data['sales_growth_5y']:.1f}%" if data.get("sales_growth_5y") else "N/A",
                    "Eff. Growth (5Y damp.)": f"{scores['blended_growth']:.1f}%" if scores.get("blended_growth") else "N/A",
                })

                val_model_metrics = {
                    "PEG Ratio (ValueLens)": f"{scores['peg']:.2f}" if scores.get("peg") else "N/A",
                    "Fair P/E (ValueLens)": f"{peg_m['fair_pe']:.2f}" if peg_m.get("fair_pe") else "N/A",
                    "Median P/E (historical)": f"{hist_p['median_pe']:.2f}" if hist_p.get("median_pe") else "N/A",
                    "Hist. Fair P/E (ValueLens)": f"{hist_p['model_pe']:.2f}" if hist_p.get("model_pe") else "N/A",
                    "Historical Premium" + (" (disregarded)" if disregard_hist_premium else ""): f"{hist_p['premium']:.2f}x" if hist_p.get("premium") else "N/A",
                    "Margin of Safety": f"{round(valuation['margin_of_safety'] * 100)}%" if valuation.get("margin_of_safety") is not None else "N/A",
                    "Exit Premium": f"{round(valuation['exit_premium'] * 100)}%" if valuation.get("exit_premium") is not None else "N/A",
                }
                _nc = valuation.get("net_cash_per_share")
                if _nc is not None:
                    _nc_sign = "+" if _nc >= 0 else "-"
                    val_model_metrics["  Cash/Share"] = f"${data.get('cash_per_share', 0):,.2f}"
                    val_model_metrics["  Debt/Share"] = f"${data.get('debt_per_share', 0):,.2f}"
                    val_model_metrics["  Leases/Share"] = f"${data.get('capital_lease_per_share', 0):,.2f}"
                    val_model_metrics["Net Cash/Debt"] = f"{_nc_sign}${abs(_nc):,.2f}"
                tbl_valuation = _html_table("Valuation Model", val_model_metrics)

                breakdown_rows = []
                for key, display_label in {"peg": "PEG Ratio (ValueLens)", "eps_revisions": "EPS Revisions", "earnings_surprises": "Earnings Surprises"}.items():
                    bd = scores["breakdown"][key]
                    breakdown_rows.append({"Component": display_label, "Score": f"{bd['score']:.1f}/10", "Weight": f"{bd['weight']:.0%}"})
                tbl_scoring = _html_breakdown_table("Scoring Breakdown", breakdown_rows)

                st.markdown(
                    "<div style='display:flex;gap:1rem'>"
                    "<div style='flex:1 1 33%;min-width:0'>"
                    f"{tbl_valuation}"
                    "</div>"
                    "<div style='flex:1 1 33%;min-width:0'>"
                    f"{tbl_growth}"
                    "</div>"
                    "<div style='flex:1 1 33%;min-width:0'>"
                    f"{tbl_earnings}{tbl_scoring}"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Gate warnings
            warnings = [m for m in r["gate_messages"] if m.startswith("Warning")]
            if warnings:
                for w in warnings:
                    st.warning(w)

        except (KeyError, TypeError):
            # Stale or incompatible session state -- clear it silently
            del st.session_state["result"]

# ===== Screener tab =====
with tab_batch:
    st.subheader("Top 500 Global Stocks")
    st.caption("Only stocks passing all valuation gates are shown (positive EPS, available growth estimates).")

    if _CSV_PATH.exists():
        run_date = _read_batch_run_date()
        if run_date:
            st.caption(f"Last run: {run_date}")

    # --- Results table ---
    _batch_df = _load_batch_csv()
    if _batch_df is None:
        st.info("No batch results yet. Run batch_analyze.py from the command line.")
    else:
        # Drop legacy Status column if present
        if "Status" in _batch_df.columns:
            _batch_df = _batch_df[_batch_df["Status"] == "OK"].drop(columns=["Status"]).reset_index(drop=True)

        # Build deep-link URLs for each ticker
        _batch_df["Ticker"] = _batch_df["Ticker"].apply(
            lambda t: f"/?ticker={t}" if pd.notna(t) and t else t
        )

        display_cols = [
            "Ticker", "Name", "Score",
            "Market Cap", "Current Price",
            "Fair Value", "% vs Fair Value",
            "Fair Price", "% vs Fair Price",
        ]
        display_cols = [c for c in display_cols if c in _batch_df.columns]

        # Use # column as index so it renders narrow
        if "#" in _batch_df.columns:
            _batch_df = _batch_df.set_index("#")



        column_config = {
            "Ticker": st.column_config.LinkColumn(
                "Ticker",
                display_text=r"/\?ticker=(.+)",
            ),
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%d",
            ),
            "Current Price": st.column_config.NumberColumn(
                "Price", format="$%.2f",
            ),
            "Fair Value": st.column_config.NumberColumn(
                "Fair Value", format="$%.2f",
            ),
            "Fair Price": st.column_config.NumberColumn(
                "Fair Price", format="$%.2f",
            ),
            "% vs Fair Value": st.column_config.NumberColumn(
                "% vs FV", format="%.1f%%",
            ),
            "% vs Fair Price": st.column_config.NumberColumn(
                "% vs FP", format="%.1f%%",
            ),
        }

        styled = _batch_df[display_cols].style.apply(
            _apply_valuation_gradient, axis=None,
        )

        st.dataframe(
            styled,
            column_config=column_config,
            hide_index=False,
            height=600,
        )

# ===== Warrant Analysis tab =====
_CURRENCIES = ["USD", "EUR", "GBP", "CHF", "JPY"]

with tab_warrant:
    st.subheader("Warrant Analysis")

    with st.form("warrant_form"):
        w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns([2, 1, 2, 2, 1])
        with w_col1:
            w_warrant_price = st.number_input("Warrant Price", min_value=0.01, value=2.35, step=0.01, format="%.2f", key="w_warrant_price")
        with w_col2:
            w_warrant_ccy = st.selectbox("Warrant Currency", _CURRENCIES, index=1, key="w_warrant_ccy")
        with w_col3:
            w_strike = st.number_input("Strike Price", min_value=0.01, value=180.00, step=0.01, format="%.2f", key="w_strike")
        with w_col4:
            w_underlying = st.number_input("Underlying Price", min_value=0.01, value=185.00, step=0.01, format="%.2f", key="w_underlying")
        with w_col5:
            w_underlying_ccy = st.selectbox("Underlying Currency", _CURRENCIES, index=0, key="w_underlying_ccy")

        w_col6, w_col7, w_col8, w_col9, w_col10 = st.columns([1, 2, 1, 1, 3])
        with w_col6:
            w_ratio = st.number_input("Ratio", min_value=0.001, value=0.10, step=0.01, format="%.3f", key="w_ratio")
        with w_col7:
            _default_expiry = date.today() + timedelta(days=365)
            w_expiry = st.date_input("Expiry Date", value=_default_expiry, min_value=date.today(), key="w_expiry")
        with w_col8:
            w_rfr = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1, format="%.1f", key="w_rfr")
        with w_col9:
            w_option_type = st.selectbox("Type", ["Call", "Put"], index=0, key="w_option_type")
        with w_col10:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            w_submitted = st.form_submit_button("Calculate", use_container_width=True, type="primary")

    # --- FX manual fallback — outside form so it persists across reruns ---
    if w_underlying_ccy != w_warrant_ccy and st.session_state.get("w_fx_fallback"):
        st.warning(f"Could not fetch FX rate for {w_underlying_ccy}/{w_warrant_ccy}. Enter manually:")
        w_manual_fx = st.number_input(
            f"FX Rate (1 {w_underlying_ccy} = ? {w_warrant_ccy})",
            min_value=0.0001, value=1.0, step=0.0001, format="%.4f",
            key="w_manual_fx",
        )
    else:
        w_manual_fx = None

    # --- Calculation (on submit or FX manual entry) ---
    _fx_ready = w_manual_fx is not None and w_manual_fx > 0 and st.session_state.get("w_fx_fallback")
    if w_submitted or _fx_ready:
        today = date.today()
        if w_expiry <= today:
            st.error("Expiry date must be in the future.")
        elif w_underlying_ccy != w_warrant_ccy and _fetch_fx_rate(w_underlying_ccy, w_warrant_ccy) is None and (w_manual_fx is None or w_manual_fx <= 0):
            st.session_state["w_fx_fallback"] = True
            st.error("Could not fetch FX rate. Please enter it manually above.")
        else:
            T = (w_expiry - today).days / 365.0
            r = w_rfr / 100.0
            opt = w_option_type.lower()

            # FX handling
            if w_underlying_ccy != w_warrant_ccy:
                fx_rate = _fetch_fx_rate(w_underlying_ccy, w_warrant_ccy)
                if fx_rate is None:
                    fx_rate = w_manual_fx
                st.session_state["w_fx_fallback"] = False
            else:
                fx_rate = None
                st.session_state.pop("w_fx_fallback", None)

            # Convert warrant price to underlying currency for IV solve
            if fx_rate is not None and fx_rate != 1.0:
                warrant_price_underlying = w_warrant_price / fx_rate
            else:
                warrant_price_underlying = w_warrant_price
                fx_rate = 1.0

            # Solve IV
            try:
                iv = implied_volatility(
                    warrant_price_underlying, w_underlying, w_strike, T, r, opt, w_ratio
                )
            except ValueError as e:
                st.error(str(e))
                iv = None

            if iv is not None:
                # Store in session state for slider reactivity
                days_to_expiry = (w_expiry - today).days
                st.session_state["w_result"] = {
                    "iv": iv,
                    "S": w_underlying,
                    "K": w_strike,
                    "T": T,
                    "r": r,
                    "option_type": opt,
                    "ratio": w_ratio,
                    "fx_rate": fx_rate,
                    "warrant_price": w_warrant_price,
                    "underlying_ccy": w_underlying_ccy,
                    "warrant_ccy": w_warrant_ccy,
                    "expiry": w_expiry,
                    "days_to_expiry": days_to_expiry,
                }
                # Explicitly reset Price Explorer sliders to defaults
                st.session_state["w_exit_price"] = round(w_underlying, 2)
                st.session_state["w_exit_days"] = 0
                st.session_state["w_iv_slider"] = round(iv * 100, 1)
                st.session_state["w_calc_count"] = st.session_state.get("w_calc_count", 0) + 1

    # --- Results display ---
    if st.session_state.get("w_result"):
        wr = st.session_state["w_result"]

        # --- Warrant Analytics ---
        iv_pct = wr["iv"] * 100
        sigma = wr["iv"]

        st.subheader("Analytics")
        g = greeks(wr["S"], wr["K"], wr["T"], wr["r"], sigma, wr["option_type"], wr["ratio"])

        # Effective leverage (omega): gear × delta
        warrant_price_ul = wr["warrant_price"] / wr["fx_rate"]
        gear = (wr["S"] / warrant_price_ul) * wr["ratio"] if warrant_price_ul > 0 else 0
        eff_leverage = abs(gear * g["delta"])

        # Breakeven at expiry: strike + cost per unit (call) or strike - cost per unit (put)
        cost_per_unit = warrant_price_ul / wr["ratio"] if wr["ratio"] > 0 else 0
        if wr["option_type"] == "call":
            breakeven = wr["K"] + cost_per_unit
        else:
            breakeven = wr["K"] - cost_per_unit

        ccy_sym = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "CHF": "CHF ", "JPY": "\u00a5"}.get(
            wr["underlying_ccy"], ""
        )

        metric_configs = [
            ("IV", iv_pct, "%", "#f59e0b", "%.1f"),
            ("Leverage", eff_leverage, "\u00d7 multiplier", "#f97316", "%.2f"),
            ("Breakeven", breakeven, f"{ccy_sym} at expiry", "#059669", "%.2f"),
            ("Delta (\u0394)", g["delta"], "price sensitivity", "#14b8a6", "%.4f"),
            ("Gamma (\u0393)", g["gamma"], "\u0394 change per $1", "#8b5cf6", "%.4f"),
            ("Theta (\u0398)", g["theta"], "per day", "#ef4444", "%.4f"),
            ("Vega (\u03bd)", g["vega"], "per 1% vol", "#3b82f6", "%.4f"),
            ("Rho (\u03c1)", g["rho"], "per 1% rate", "#06b6d4", "%.4f"),
        ]

        metric_cards = ""
        for name, val, unit, color, fmt in metric_configs:
            _n = html_mod.escape(name)
            _u = html_mod.escape(unit)
            _v = fmt % val
            metric_cards += (
                f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;"
                f"padding:16px;text-align:center;flex:1'>"
                f"<div style='font-size:0.75rem;color:rgba(224,228,238,0.6);margin-bottom:6px'>{_n}</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:{color}'>{_v}</div>"
                f"<div style='font-size:0.7rem;color:rgba(224,228,238,0.4);margin-top:4px'>{_u}</div>"
                f"</div>"
            )

        st.markdown(
            f"<div style='display:flex;gap:12px'>{metric_cards}</div>",
            unsafe_allow_html=True,
        )

        # --- Scenario Analysis (fragment for instant slider updates) ---
        @st.fragment
        def _scenario_fragment(calc_count: int):
            _wr = st.session_state["w_result"]
            _iv_pct = _wr["iv"] * 100
            _days_to_expiry = _wr["days_to_expiry"]

            # Initialize slider defaults in session state (first run only)
            if "w_exit_price" not in st.session_state:
                st.session_state["w_exit_price"] = round(_wr["S"], 2)
            if "w_exit_days" not in st.session_state:
                st.session_state["w_exit_days"] = 0
            if "w_iv_slider" not in st.session_state:
                st.session_state["w_iv_slider"] = round(_iv_pct, 1)

            st.markdown("")  # spacing to match other sections
            st.subheader("Price Explorer")
            sc_left, sc_right = st.columns([3, 2])

            with sc_left:
                exit_price_min = round(_wr["S"] * 0.5, 2)
                exit_price_max = round(_wr["S"] * 1.5, 2)
                _exit_price = st.slider(
                    f"Exit Underlying Price ({_wr['underlying_ccy']})",
                    min_value=exit_price_min,
                    max_value=exit_price_max,
                    step=0.01,
                    format="%.2f",
                    key="w_exit_price",
                )
                _exit_days = st.slider(
                    "Days to Expiry",
                    min_value=0,
                    max_value=max(_days_to_expiry, 1),
                    step=1,
                    key="w_exit_days",
                )
                _iv_slider = st.slider(
                    "Implied Volatility",
                    min_value=1.0, max_value=200.0,
                    step=0.1,
                    format="%.1f%%",
                    key="w_iv_slider",
                )

            with sc_right:
                # Compute scenario inside the column context so rendering stays side-by-side
                _sigma = _iv_slider / 100.0
                T_exit = max(_exit_days / 365.0, 0)

                projected = scenario_price(
                    _exit_price, T_exit, _wr["r"], _sigma, _wr["K"], _wr["option_type"], _wr["ratio"]
                )

                projected_warrant_ccy = round(projected * _wr["fx_rate"], 2)
                appreciation = ((projected_warrant_ccy - _wr["warrant_price"]) / _wr["warrant_price"]) * 100
                appreciation = round(appreciation, 1)
                if appreciation == 0.0:
                    appreciation = 0.0  # normalize -0.0 to 0.0

                ccy_symbol = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "CHF": "CHF ", "JPY": "\u00a5"}.get(
                    _wr["warrant_ccy"], ""
                )
                appr_color = "#10b981" if appreciation > 0 else "#ef4444" if appreciation < 0 else "rgba(224,228,238,0.7)"
                appr_sign = "+" if appreciation > 0 else ""
                arrow = "&#9650;" if appreciation > 0 else "&#9660;" if appreciation < 0 else "&#9679;"

                # Result cards — single HTML block to avoid extra Streamlit spacing
                _appr_rgb = '16,185,129' if appreciation >= 0 else '239,68,68'
                st.markdown(
                    f"<div style='display:flex;flex-direction:column;gap:12px'>"
                    # Projected price card
                    f"<div style='background:rgba(255,255,255,0.06);border-radius:12px;"
                    f"padding:20px;text-align:center;"
                    f"border:1px solid rgba(255,255,255,0.08)'>"
                    f"<div style='font-size:0.7rem;color:rgba(224,228,238,0.5);text-transform:uppercase;"
                    f"letter-spacing:1px;margin-bottom:6px'>Projected Price</div>"
                    f"<div style='font-size:2rem;font-weight:700;letter-spacing:-0.5px'>"
                    f"{ccy_symbol}{projected_warrant_ccy:,.2f}</div>"
                    f"</div>"
                    # Appreciation card
                    f"<div style='background:rgba({_appr_rgb},0.08);"
                    f"border-radius:12px;padding:20px;text-align:center;"
                    f"border:1px solid rgba({_appr_rgb},0.2)'>"
                    f"<div style='font-size:0.7rem;color:rgba(224,228,238,0.5);text-transform:uppercase;"
                    f"letter-spacing:1px;margin-bottom:6px'>Appreciation</div>"
                    f"<div style='font-size:2rem;font-weight:700;color:{appr_color}'>"
                    f"{arrow} {appr_sign}{appreciation:.1f}%</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        _scenario_fragment(st.session_state.get("w_calc_count", 0))
