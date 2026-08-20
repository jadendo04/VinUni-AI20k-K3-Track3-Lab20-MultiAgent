"""Routing policy tests - the core design decision of the lab."""

from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.state import ResearchState


def test_routes_to_researcher_when_state_empty(settings: Settings, state: ResearchState) -> None:
    assert SupervisorAgent(settings=settings).decide(state) == "researcher"


def test_routes_to_analyst_after_research(settings: Settings, state: ResearchState) -> None:
    state.research_notes = "notes"
    assert SupervisorAgent(settings=settings).decide(state) == "analyst"


def test_routes_to_writer_after_analysis(settings: Settings, state: ResearchState) -> None:
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    assert SupervisorAgent(settings=settings).decide(state) == "writer"


def test_stops_when_answer_exists(settings: Settings, state: ResearchState) -> None:
    state.research_notes = "notes"
    state.final_answer = "answer"
    assert SupervisorAgent(settings=settings).decide(state) == DONE


def test_runs_critic_once_when_enabled(settings: Settings, state: ResearchState) -> None:
    state.final_answer = "answer"
    supervisor = SupervisorAgent(settings=settings, enable_critic=True)
    assert supervisor.decide(state) == "critic"
    state.record_route("critic")
    assert supervisor.decide(state) == DONE


def test_iteration_cap_stops_the_loop(settings: Settings, state: ResearchState) -> None:
    supervisor = SupervisorAgent(settings=settings, max_iterations=2)
    state.iteration = 2
    assert supervisor.decide(state) == DONE


def test_failing_agent_is_skipped_not_retried_forever(
    settings: Settings, state: ResearchState
) -> None:
    supervisor = SupervisorAgent(settings=settings, max_agent_retries=2)
    state.record_failure("researcher", "boom")
    assert supervisor.decide(state) == "researcher"
    state.record_failure("researcher", "boom")
    assert supervisor.decide(state) == DONE


def test_run_records_route_and_trace(settings: Settings, state: ResearchState) -> None:
    result = SupervisorAgent(settings=settings).run(state)
    assert result.route_history == ["researcher"]
    assert result.next_route == "researcher"
    assert result.trace[0]["name"] == "supervisor.route"
