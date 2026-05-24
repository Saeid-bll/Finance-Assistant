"""Goal planning view."""

from __future__ import annotations

from typing import Any

from agents.goals import GoalPlanningAgent


def goal_payload(
    *,
    name: str,
    target_amount: float,
    current_amount: float,
    monthly_contribution: float,
    time_horizon_years: float,
    risk_appetite: str,
) -> dict[str, Any]:
    """Create an agent-ready financial goal payload."""

    return {
        "name": name.strip(),
        "target_amount": target_amount,
        "current_amount": current_amount,
        "monthly_contribution": monthly_contribution,
        "time_horizon_years": time_horizon_years,
        "risk_appetite": risk_appetite,
    }


def render_goals_view() -> None:
    """Render the Streamlit goals tab."""

    import streamlit as st

    st.subheader("Goals")
    with st.form("goal_form"):
        name = st.text_input("Goal", value="House down payment")
        target_amount = st.number_input("Target amount", min_value=1.0, value=50000.0, step=1000.0)
        current_amount = st.number_input("Current amount", min_value=0.0, value=5000.0, step=500.0)
        monthly_contribution = st.number_input(
            "Monthly contribution", min_value=0.0, value=750.0, step=50.0
        )
        time_horizon_years = st.number_input("Years", min_value=0.1, value=5.0, step=0.5)
        risk_appetite = st.selectbox("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
        submitted = st.form_submit_button("Project goal", type="primary", use_container_width=True)

    if submitted:
        payload = goal_payload(
            name=name,
            target_amount=target_amount,
            current_amount=current_amount,
            monthly_contribution=monthly_contribution,
            time_horizon_years=time_horizon_years,
            risk_appetite=risk_appetite,
        )
        st.session_state["goal_response"] = GoalPlanningAgent().run({"goal": payload})

    response = st.session_state.get("goal_response")
    if response is None:
        return
    if response.error_code:
        st.error(response.content)
        return

    projection = response.metadata.get("projection", {})
    projected_balance = projection.get("projected_balance", 0)
    shortfall_or_surplus = projection.get("shortfall_or_surplus", 0)
    expected_return = projection.get("expected_return", 0)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Projected balance", f"${projected_balance:,.2f}")
    metric_cols[1].metric("Shortfall or surplus", f"${shortfall_or_surplus:,.2f}")
    metric_cols[2].metric("Return assumption", f"{expected_return:.1%}")
    st.markdown(response.content)

    assumptions = projection.get("assumptions") or []
    if assumptions:
        with st.expander("Assumptions", expanded=False):
            for assumption in assumptions:
                st.markdown(f"- {assumption}")
