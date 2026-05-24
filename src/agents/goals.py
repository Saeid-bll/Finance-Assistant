"""Goal planning agent."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agents.base import BaseAgent
from core.disclaimers import disclaimer_text
from core.models import AgentResponse, FinancialGoal, GoalProjection


EXPECTED_RETURNS = {
    "conservative": 0.03,
    "moderate": 0.05,
    "aggressive": 0.07,
}


class GoalPlanningAgent(BaseAgent):
    """Project goal progress with simple assumption-based math."""

    agent_name = "goals"

    def project(self, goal: dict[str, Any] | FinancialGoal) -> GoalProjection:
        parsed_goal = goal if isinstance(goal, FinancialGoal) else FinancialGoal.model_validate(goal)
        expected_return = EXPECTED_RETURNS[parsed_goal.risk_appetite]
        projected_balance = self._future_value(parsed_goal, expected_return)
        shortfall_or_surplus = projected_balance - parsed_goal.target_amount
        assumptions = [
            f"Uses a {expected_return:.1%} annual return assumption for a {parsed_goal.risk_appetite} profile.",
            "Assumes monthly contributions are made consistently.",
            "Does not include taxes, fees, inflation, or market volatility.",
        ]
        educational_summary = self._summary(parsed_goal, projected_balance, shortfall_or_surplus)

        return GoalProjection(
            name=parsed_goal.name,
            target_amount=parsed_goal.target_amount,
            projected_balance=projected_balance,
            shortfall_or_surplus=shortfall_or_surplus,
            expected_return=expected_return,
            assumptions=assumptions,
            educational_summary=educational_summary,
            disclaimer=disclaimer_text(),
        )

    def run(self, payload: Any) -> AgentResponse:
        try:
            goal = payload.get("goal", payload) if isinstance(payload, dict) else payload
            projection = self.project(goal)
        except (TypeError, ValueError, ValidationError) as exc:
            return self.response(
                f"I could not project this goal because the input looks invalid: {exc}",
                confidence=0.2,
                error_code="INVALID_GOAL",
            )

        return self.response(
            projection.educational_summary,
            confidence=0.85,
            metadata={"projection": projection.model_dump()},
        )

    def _future_value(self, goal: FinancialGoal, annual_return: float) -> float:
        months = int(round(goal.time_horizon_years * 12))
        monthly_return = annual_return / 12
        if months <= 0:
            return goal.current_amount
        if monthly_return == 0:
            return goal.current_amount + goal.monthly_contribution * months

        growth_factor = (1 + monthly_return) ** months
        current_value = goal.current_amount * growth_factor
        contribution_value = goal.monthly_contribution * ((growth_factor - 1) / monthly_return)
        return current_value + contribution_value

    def _summary(
        self,
        goal: FinancialGoal,
        projected_balance: float,
        shortfall_or_surplus: float,
    ) -> str:
        base = (
            f"{goal.name} is projected to reach ${projected_balance:,.2f} "
            f"against a ${goal.target_amount:,.2f} target. "
            "This projection is an assumption-based estimate, not a promise of actual results."
        )
        if shortfall_or_surplus < 0:
            return (
                f"{base} The estimate shows a shortfall of ${abs(shortfall_or_surplus):,.2f}; "
                "consider whether you can increase contributions, extend the timeline, or revisit the target."
            )
        return f"{base} The estimate shows a surplus of ${shortfall_or_surplus:,.2f}."
