import statistics
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import calculate_valuation, compute_historical_pe_series


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch(ticker: str) -> dict:
    """Cache yfinance data for 15 minutes to avoid rate limits."""
    return fetch_stock_data(ticker)


# --- Page config ---
st.set_page_config(page_title="Stock Analyzer", layout="centered")
st.title("Stock Fundamental Analyzer")


# --- Section 1: Input bar ---
with st.form("analyze_form"):
    col_ticker, col_button = st.columns([3, 1])
    with col_ticker:
        ticker = st.text_input("Ticker", placeholder="e.g. NVDA", label_visibility="collapsed")
    with col_button:
        analyze = st.form_submit_button("Analyze", type="primary", use_container_width=True)

with st.expander("Advanced Options"):
    use_custom_eps = st.checkbox("Override trailing EPS")
    custom_eps = None
    if use_custom_eps:
        custom_eps = st.number_input(
            "Custom EPS",
            min_value=0.01,
            value=1.00,
            step=0.01,
            format="%.2f",
            placeholder="e.g. 5.67",
            help="Override trailing EPS for stocks with GAAP distortions",
        )
    use_custom_growth = st.checkbox("Override 5Y growth rate")
    custom_growth = None
    if use_custom_growth:
        custom_growth = st.number_input(
            "Custom 5Y Growth (%)",
            min_value=0.1,
            value=15.0,
            step=0.5,
            format="%.1f",
            placeholder="e.g. 15.0",
            help="Override 5Y growth rate. Bypasses blended growth calculation.",
        )


# --- Orchestration ---
if analyze and ticker:
    ticker = ticker.upper().strip()

    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            data = cached_fetch(ticker)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            st.stop()

    passed, gate_messages = check_gates(data)

    if not passed:
        st.error(f"**{ticker} — {data.get('name', 'Unknown')}**")
        for msg in gate_messages:
            st.error(msg)
        st.stop()

    scores = score_stock(data, custom_eps=custom_eps, custom_growth=custom_growth)
    valuation = calculate_valuation(
        data, custom_eps=custom_eps, custom_growth=custom_growth, scores=scores
    )
    scores = apply_price_cap(scores, data, valuation)
    pe_series = compute_historical_pe_series(data)

    st.session_state["result"] = {
        "ticker": ticker,
        "data": data,
        "scores": scores,
        "valuation": valuation,
        "pe_series": pe_series,
        "gate_messages": gate_messages,
    }


# --- Rendering (from session state) ---
if "result" in st.session_state:
    r = st.session_state["result"]
    ticker = r["ticker"]
    data = r["data"]
    scores = r["scores"]
    valuation = r["valuation"]
    pe_series = r["pe_series"]

    # --- Section 2: Score & Summary ---
    name = data.get("name", ticker)
    price = data.get("current_price")

    website = data.get("website")
    logo_url = ""
    if website:
        domain = urlparse(website).netloc or urlparse(website).path
        logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"

    if logo_url:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px'>"
            f"<img src='{logo_url}' width='40' height='40' style='border-radius:6px'"
            f" onerror=\"this.style.display='none'\">"
            f"<span style='font-size:1.5em;font-weight:600'>{ticker} — {name}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"### {ticker} — {name}")
    if price:
        st.markdown(f"**Current Price:** ${price:,.2f}")

    score = scores["final_score"]
    label = scores["label"]

    score_colors = {
        "Strong Buy": "green", "Attractive": "green",
        "Hold / Fair Value": "orange", "Unattractive": "red", "Avoid": "red",
    }
    score_color = score_colors.get(label, "gray")
    st.markdown(
        f"<h2 style='color:{score_color};margin-top:0'>{score}/100 — {label}</h2>",
        unsafe_allow_html=True,
    )

    # Metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        fv = valuation.get("fair_value")
        st.metric("Fair Value", f"${fv:,.2f}" if fv else "N/A")
    with col2:
        ep = valuation.get("entry_price")
        margin_pct = round(valuation.get("margin_of_safety", 0) * 100)
        st.metric(
            "Entry Price",
            f"<= ${ep:,.2f}" if ep else "N/A",
            delta=f"{margin_pct}% margin",
            delta_color="off",
        )
    with col3:
        xp = valuation.get("exit_price")
        exit_pct = round(valuation.get("exit_premium", 0) * 100)
        st.metric(
            "Exit Price",
            f">= ${xp:,.2f}" if xp else "N/A",
            delta=f"{exit_pct}% premium",
            delta_color="off",
        )

    if scores.get("peg") is not None:
        st.markdown(f"**PEG Ratio (blended):** {scores['peg']:.2f}")

    st.divider()

    # --- Section 3: Price chart ---
    historical_prices = data.get("historical_prices")
    if historical_prices:
        dates = [p["date"] for p in historical_prices]
        prices = [p["price"] for p in historical_prices]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines",
            name="Price", line=dict(color="#2196F3", width=2),
        ))

        fv = valuation.get("fair_value")
        ep = valuation.get("entry_price")
        xp = valuation.get("exit_price")

        if fv:
            fig.add_hline(
                y=fv, line_dash="solid", line_color="blue",
                annotation_text=f"Fair Value ${fv:,.0f}",
            )
        if ep:
            fig.add_hline(
                y=ep, line_dash="dash", line_color="green",
                annotation_text=f"Entry ${ep:,.0f}",
            )
        if xp:
            fig.add_hline(
                y=xp, line_dash="dash", line_color="red",
                annotation_text=f"Exit ${xp:,.0f}",
            )

        fig.update_layout(
            title=f"{ticker} — Price vs Fair Value",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Section 4: Historical P/E chart ---
    if pe_series:
        pe_dates = [p["date"] for p in pe_series]
        pe_values = [p["pe"] for p in pe_series]
        median_pe = statistics.median(pe_values)

        fig_pe = go.Figure()
        fig_pe.add_trace(go.Scatter(
            x=pe_dates, y=pe_values, mode="lines",
            name="P/E", line=dict(color="#9C27B0", width=2),
        ))
        fig_pe.add_hline(
            y=median_pe, line_dash="dash", line_color="gray",
            annotation_text=f"Median P/E {median_pe:.1f}",
        )

        current_pe = data.get("trailing_pe")
        if current_pe:
            fig_pe.add_hline(
                y=current_pe, line_dash="dot", line_color="orange",
                annotation_text=f"Current P/E {current_pe:.1f}",
                annotation_position="bottom right",
            )

        fig_pe.update_layout(
            title=f"{ticker} — Historical P/E",
            xaxis_title="Date",
            yaxis_title="P/E Ratio",
            height=350,
            showlegend=False,
        )

        st.plotly_chart(fig_pe, use_container_width=True)

    # --- Section 5: Detailed Metrics (expandable) ---
    with st.expander("Detailed Metrics"):

        # Force consistent column widths across metric tables
        st.markdown(
            "<style>"
            "[data-testid='stExpander'] table { table-layout: fixed; width: 100%; }"
            "[data-testid='stExpander'] table th:first-child,"
            "[data-testid='stExpander'] table td:first-child { width: 65%; }"
            "table.val-breakdown { table-layout: fixed; width: 100%; }"
            "table.val-breakdown th:first-child,"
            "table.val-breakdown td:first-child { width: 40%; }"
            "table.val-breakdown th:nth-child(2),"
            "table.val-breakdown td:nth-child(2) { width: 25%; }"
            "table.val-breakdown th:nth-child(3),"
            "table.val-breakdown td:nth-child(3) { width: 35%; }"
            "</style>",
            unsafe_allow_html=True,
        )

        peg_m = valuation.get("peg_method", {})
        hist_m = valuation.get("historical_method", {})

        def _metrics_table(metrics: dict) -> None:
            df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).set_index("Metric")
            st.table(df)

        st.markdown("**Earnings & Price**")
        _metrics_table({
            "Trailing EPS": f"${data['trailing_eps']:.2f}" if data.get("trailing_eps") else "N/A",
            "Trailing P/E": f"{data['trailing_pe']:.2f}" if data.get("trailing_pe") else "N/A",
            "Forward P/E": f"{data['forward_pe']:.2f}" if data.get("forward_pe") else "N/A",
            "P/S Ratio": f"{data['ps_ratio']:.2f}" if data.get("ps_ratio") else "N/A",
            "Beta": f"{data['beta']:.2f}" if data.get("beta") else "N/A",
        })

        st.markdown("**Growth**")
        _metrics_table({
            "0Y EPS Growth": f"{data['growth_current_year']:.1f}%" if data.get("growth_current_year") else "N/A",
            "1Y EPS Growth": f"{data['growth_next_year']:.1f}%" if data.get("growth_next_year") else "N/A",
            "5Y Est. Growth": f"{data['growth_5y']:.1f}%" if data.get("growth_5y") else "N/A",
            "Historical CAGR": f"{data['historical_growth_5y']:.1f}%" if data.get("historical_growth_5y") else "N/A",
            "Blended Growth": f"{scores['blended_growth']:.1f}%" if scores.get("blended_growth") else "N/A",
        })

        st.markdown("**Valuation Ratios**")
        _metrics_table({
            "PEG (blended)": f"{scores['peg']:.2f}" if scores.get("peg") else "N/A",
            "PSG Ratio": f"{scores['psg']:.2f}" if scores.get("psg") else "N/A",
        })

        st.markdown("**Valuation Model**")
        _metrics_table({
            "Fair P/E (PEG-implied)": f"{peg_m['fair_pe']:.2f}" if peg_m.get("fair_pe") else "N/A",
            "Median P/E (historical)": f"{hist_m['median_pe']:.2f}" if hist_m.get("median_pe") else "N/A",
            "Growth Ratio": f"{hist_m['growth_ratio']:.2f}x" if hist_m.get("growth_ratio") else "N/A",
            "Adjusted P/E (historical)": f"{hist_m['adjusted_pe']:.2f}" if hist_m.get("adjusted_pe") else "N/A",
            "Margin of Safety": f"{round(valuation['margin_of_safety'] * 100)}%" if valuation.get("margin_of_safety") is not None else "N/A",
            "Exit Premium": f"{round(valuation['exit_premium'] * 100)}%" if valuation.get("exit_premium") is not None else "N/A",
        })

        st.markdown("**Scoring Breakdown**")
        breakdown_labels = {
            "peg": "PEG Ratio (blended)",
            "psg": "PSG Ratio",
            "eps_revisions": "EPS Revisions",
            "earnings_surprises": "Earnings Surprises",
        }
        breakdown_rows = []
        for key, display_label in breakdown_labels.items():
            bd = scores["breakdown"][key]
            breakdown_rows.append({
                "Component": display_label,
                "Score": f"{bd['score']:.1f}/10",
                "Weight": f"{bd['weight']:.0%}",
            })
        breakdown_df = pd.DataFrame(breakdown_rows).set_index("Component")
        st.table(breakdown_df)

        st.markdown("**Valuation Breakdown**")
        peg_mv = valuation["peg_method"]
        hist_mv = valuation["historical_method"]
        val_rows = []

        if peg_mv.get("fair_price"):
            val_rows.append({
                "Method": "PEG-Implied (60%)",
                "Fair Price": f"${peg_mv['fair_price']:,.2f}",
                "Detail": f"Fair P/E = {peg_mv['fair_pe']}",
            })
        if hist_mv.get("fair_price"):
            detail = f"Median P/E = {hist_mv['median_pe']}"
            if hist_mv.get("growth_ratio") and hist_mv["growth_ratio"] != 1.0:
                detail += f", growth adj = {hist_mv['growth_ratio']}x"
            val_rows.append({
                "Method": "Historical Adj. (40%)",
                "Fair Price": f"${hist_mv['fair_price']:,.2f}",
                "Detail": detail,
            })
        if valuation.get("fair_value"):
            val_rows.append({
                "Method": "Blended Fair Value",
                "Fair Price": f"${valuation['fair_value']:,.2f}",
                "Detail": "",
            })

        if val_rows:
            html = "<table class='val-breakdown'>"
            html += "<thead><tr><th>Method</th><th>Fair Price</th><th>Detail</th></tr></thead><tbody>"
            for r_ in val_rows:
                html += f"<tr><td>{r_['Method']}</td><td>{r_['Fair Price']}</td><td>{r_['Detail']}</td></tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)

    # Gate warnings
    warnings = [m for m in r["gate_messages"] if m.startswith("Warning")]
    if warnings:
        for w in warnings:
            st.warning(w)
