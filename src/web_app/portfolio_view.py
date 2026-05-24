"""Portfolio dashboard view."""

from __future__ import annotations

from typing import Any

from agents.portfolio import PortfolioAnalysisAgent


DEFAULT_HOLDINGS = [
    {"ticker": "VTI", "quantity": 10.0, "price": 250.0, "asset_type": "ETF"},
    {"ticker": "VXUS", "quantity": 20.0, "price": 60.0, "asset_type": "ETF"},
    {"ticker": "BND", "quantity": 15.0, "price": 72.0, "asset_type": "Bond ETF"},
]


def holding_rows_to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert editable table rows into agent-ready holding records."""

    records = []
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        records.append(
            {
                "ticker": ticker,
                "quantity": float(row.get("quantity") or 0),
                "price": float(row.get("price") or 0),
                "asset_type": str(row.get("asset_type") or "Unknown").strip() or "Unknown",
            }
        )
    return records


def render_portfolio_view() -> None:
    """Render the Streamlit portfolio tab."""

    import pandas as pd
    import streamlit as st

    st.subheader("Portfolio")
    rows = st.data_editor(
        st.session_state.setdefault("portfolio_rows", DEFAULT_HOLDINGS),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=1.0),
            "price": st.column_config.NumberColumn("Price", min_value=0.0, step=1.0, format="$%.2f"),
            "asset_type": st.column_config.TextColumn("Asset type"),
        },
        key="portfolio_editor",
    )

    if st.button("Analyze portfolio", type="primary", use_container_width=True):
        response = PortfolioAnalysisAgent().run({"holdings": holding_rows_to_records(rows)})
        st.session_state["portfolio_response"] = response

    response = st.session_state.get("portfolio_response")
    if response is None:
        return
    if response.error_code:
        st.error(response.content)
        return

    analysis = response.metadata.get("analysis", {})
    total_value = analysis.get("total_value", 0)
    diversification_score = analysis.get("diversification_score", 0)
    allocations = analysis.get("allocations", {})

    metric_cols = st.columns(2)
    metric_cols[0].metric("Total value", f"${total_value:,.2f}")
    metric_cols[1].metric("Diversification score", f"{diversification_score:.0f}/100")
    st.markdown(response.content)

    if allocations:
        chart_data = pd.DataFrame(
            [{"Ticker": ticker, "Allocation": allocation} for ticker, allocation in allocations.items()]
        ).set_index("Ticker")
        st.bar_chart(chart_data, use_container_width=True)

    warnings = analysis.get("concentration_warnings") or []
    for warning in warnings:
        st.warning(warning)
