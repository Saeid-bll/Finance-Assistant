import pytest


pytestmark = [pytest.mark.unit, pytest.mark.contract]
pending = pytest.mark.xfail(strict=True, reason="Pending goal planning agent implementation")


@pending
def test_goal_agent_projects_future_balance(require_attr, sample_goal) -> None:
    GoalPlanningAgent = require_attr("agents.goals", "GoalPlanningAgent")
    agent = GoalPlanningAgent()

    result = agent.project(sample_goal)

    assert result.projected_balance > sample_goal["current_amount"]
    assert result.assumptions


@pending
def test_goal_agent_reports_shortfall_or_surplus(require_attr, sample_goal) -> None:
    GoalPlanningAgent = require_attr("agents.goals", "GoalPlanningAgent")
    agent = GoalPlanningAgent()

    result = agent.project(sample_goal)

    assert result.target_amount == sample_goal["target_amount"]
    assert result.shortfall_or_surplus == pytest.approx(
        result.projected_balance - sample_goal["target_amount"]
    )


@pending
def test_goal_agent_varies_assumptions_by_risk_appetite(require_attr, sample_goal) -> None:
    GoalPlanningAgent = require_attr("agents.goals", "GoalPlanningAgent")
    agent = GoalPlanningAgent()

    conservative = agent.project({**sample_goal, "risk_appetite": "conservative"})
    aggressive = agent.project({**sample_goal, "risk_appetite": "aggressive"})

    assert conservative.expected_return < aggressive.expected_return


@pending
def test_goal_agent_rejects_negative_contribution(require_attr, sample_goal) -> None:
    GoalPlanningAgent = require_attr("agents.goals", "GoalPlanningAgent")
    agent = GoalPlanningAgent()

    with pytest.raises(ValueError):
        agent.project({**sample_goal, "monthly_contribution": -100})


@pending
def test_goal_agent_explains_projection_is_assumption_based(require_attr, sample_goal) -> None:
    GoalPlanningAgent = require_attr("agents.goals", "GoalPlanningAgent")
    agent = GoalPlanningAgent()

    result = agent.project(sample_goal)

    assert "assumption" in result.educational_summary.lower()
    assert "not financial advice" in result.disclaimer.lower()

