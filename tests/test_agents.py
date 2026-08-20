"""Worker agent contract tests."""

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.baseline import SingleAgentBaseline
from multi_agent_research_lab.agents.critic import CriticAgent, citation_coverage
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState


def test_researcher_fills_sources_and_notes(
    state: ResearchState, llm: object, search: object
) -> None:
    result = ResearcherAgent(llm=llm, search=search).run(state)
    assert result.sources
    assert result.research_notes
    assert result.agent_results[0].metadata["source_count"] == len(result.sources)


def test_analyst_requires_research_notes(state: ResearchState, llm: object) -> None:
    with pytest.raises(AgentExecutionError):
        AnalystAgent(llm=llm).run(state)


def test_writer_appends_reference_list(state: ResearchState, llm: object, search: object) -> None:
    state = ResearcherAgent(llm=llm, search=search).run(state)
    state = AnalystAgent(llm=llm).run(state)
    state = WriterAgent(llm=llm).run(state)
    assert state.final_answer is not None
    assert "## Sources" in state.final_answer


def test_execute_captures_failure_instead_of_raising(state: ResearchState, llm: object) -> None:
    result = AnalystAgent(llm=llm).execute(state)
    assert result.failure_count("analyst") == 1
    assert result.errors and result.errors[0].startswith("analyst:")


def test_critic_flags_missing_citations(state: ResearchState) -> None:
    state.final_answer = "short answer without citations"
    state.sources = []
    result = CriticAgent().run(state)
    assert result.agent_results[-1].metadata["verdict"] == "needs-revision"


def test_citation_coverage_counts_distinct_valid_markers() -> None:
    assert citation_coverage("claim [1] and [2] and [2]", 4) == pytest.approx(0.5)
    assert citation_coverage("claim [9]", 2) == 0.0
    assert citation_coverage("no markers", 0) == 0.0


def test_baseline_produces_answer_in_one_llm_call(
    state: ResearchState, baseline_agent: SingleAgentBaseline
) -> None:
    result = baseline_agent.run(state)
    assert result.final_answer
    assert result.llm_calls == 1
    assert result.route_history == ["baseline"]
