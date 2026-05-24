import pytest


pytestmark = [pytest.mark.integration, pytest.mark.contract]


def test_goal_flow_validates_projects_and_formats_summary(require_attr, sample_goal) -> None:
    GoalPlanningAgent = require_attr("agents.goals", "GoalPlanningAgent")
    format_goal_summary = require_attr("utils.formatting", "format_goal_summary")

    agent = GoalPlanningAgent()
    projection = agent.project(sample_goal)
    summary = format_goal_summary(projection)

    assert projection.projected_balance > 0
    assert sample_goal["name"] in summary
    assert "assumption" in summary.lower()


def test_goal_flow_handles_unreachable_goal_conservatively(require_attr, sample_goal) -> None:
    GoalPlanningAgent = require_attr("agents.goals", "GoalPlanningAgent")
    agent = GoalPlanningAgent()

    projection = agent.project(
        {
            **sample_goal,
            "target_amount": 1_000_000,
            "monthly_contribution": 50,
            "time_horizon_years": 2,
        }
    )

    assert projection.shortfall_or_surplus < 0
    assert "increase contributions" in projection.educational_summary.lower()
    assert "guarantee" not in projection.educational_summary.lower()
