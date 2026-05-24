"""Market overview view."""

from __future__ import annotations

from agents.market import MarketAnalysisAgent


def render_market_view() -> None:
    """Render the Streamlit market lookup tab."""

    import streamlit as st

    st.subheader("Market")
    ticker = st.text_input("Ticker", value="VTI", max_chars=12).strip().upper()

    if st.button("Look up quote", type="primary", use_container_width=True):
        st.session_state["market_response"] = MarketAnalysisAgent().run({"ticker": ticker})

    response = st.session_state.get("market_response")
    if response is None:
        return
    if response.error_code:
        st.error(response.content)
        return

    quote = response.metadata.get("quote", {})
    price = quote.get("price")
    currency = quote.get("currency", "USD")
    as_of = quote.get("as_of", "")
    provider = quote.get("provider", "")

    if price is not None:
        st.metric(str(quote.get("ticker") or ticker), f"{float(price):,.2f} {currency}")
    st.markdown(response.content)
    if as_of or provider:
        st.caption(f"{provider} {as_of}".strip())
