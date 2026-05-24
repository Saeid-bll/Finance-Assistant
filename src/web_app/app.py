"""Streamlit entrypoint for the AI Finance Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from web_app.chat import render_chat_view
from web_app.goals_view import render_goals_view
from web_app.market_view import render_market_view
from web_app.portfolio_view import render_portfolio_view


def main() -> None:
    st.set_page_config(page_title="AI Finance Assistant", layout="wide")
    st.title("AI Finance Assistant")

    chat_tab, portfolio_tab, market_tab, goals_tab = st.tabs(
        ["Chat", "Portfolio", "Market", "Goals"]
    )
    with chat_tab:
        render_chat_view()
    with portfolio_tab:
        render_portfolio_view()
    with market_tab:
        render_market_view()
    with goals_tab:
        render_goals_view()


if __name__ == "__main__":
    main()
