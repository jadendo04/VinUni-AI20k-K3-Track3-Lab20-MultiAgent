"""Shared fixtures: every test runs against the deterministic offline backends."""

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.baseline import SingleAgentBaseline
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

QUERY = "Compare single-agent and multi-agent workflows for customer support"


@pytest.fixture
def settings() -> Settings:
    return Settings(OPENAI_API_KEY=None, TAVILY_API_KEY=None, MAX_ITERATIONS=6)


@pytest.fixture
def llm(settings: Settings) -> LLMClient:
    return LLMClient(settings=settings, force_offline=True)


@pytest.fixture
def search(settings: Settings) -> SearchClient:
    return SearchClient(settings=settings, force_mock=True)


@pytest.fixture
def state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query=QUERY))


@pytest.fixture
def workers(llm: LLMClient, search: SearchClient) -> dict[str, object]:
    return {
        "researcher": ResearcherAgent(llm=llm, search=search),
        "analyst": AnalystAgent(llm=llm),
        "writer": WriterAgent(llm=llm),
        "critic": CriticAgent(),
    }


@pytest.fixture
def baseline_agent(llm: LLMClient, search: SearchClient) -> SingleAgentBaseline:
    return SingleAgentBaseline(llm=llm, search=search)
