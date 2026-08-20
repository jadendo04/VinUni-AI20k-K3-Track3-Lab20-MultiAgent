"""End-to-end workflow tests for both execution engines."""

import pytest

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import FALLBACK_PREFIX, MultiAgentWorkflow


class BrokenAgent(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        raise AgentExecutionError("search backend down")


@pytest.mark.parametrize("use_langgraph", [False, True])
def test_full_route_produces_answer(
    settings: Settings, state: ResearchState, workers: dict, use_langgraph: bool
) -> None:
    workflow = MultiAgentWorkflow(
        settings=settings, workers=workers, enable_critic=True, use_langgraph=use_langgraph
    )
    result = workflow.run(state)
    assert result.route_history == ["researcher", "analyst", "writer", "critic", "done"]
    assert result.final_answer and not result.final_answer.startswith(FALLBACK_PREFIX)
    assert result.errors == []
    assert result.trace[-1]["name"] == "workflow.completed"


def test_workflow_degrades_gracefully_when_researcher_fails(
    settings: Settings, state: ResearchState, workers: dict
) -> None:
    workers = {**workers, "researcher": BrokenAgent()}
    workflow = MultiAgentWorkflow(settings=settings, workers=workers, use_langgraph=False)
    result = workflow.run(state)
    assert result.failure_count("researcher") == 2
    assert result.final_answer is not None
    assert result.final_answer.startswith(FALLBACK_PREFIX)


def test_iteration_cap_bounds_the_run(
    settings: Settings, state: ResearchState, workers: dict
) -> None:
    capped = settings.model_copy(update={"max_iterations": 2})
    workflow = MultiAgentWorkflow(settings=capped, workers=workers, use_langgraph=False)
    result = workflow.run(state)
    dispatched = [route for route in result.route_history if route != "done"]
    assert len(dispatched) <= 2  # the cap bounds worker dispatches
    assert result.route_history[-1] == "done"
    assert result.final_answer is not None  # degraded, but never empty
