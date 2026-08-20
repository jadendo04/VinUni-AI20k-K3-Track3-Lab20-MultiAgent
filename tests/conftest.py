"""Shared fixtures: every test runs against the deterministic offline backends."""

from collections.abc import Iterator

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.baseline import SingleAgentBaseline
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

QUERY = "Compare single-agent and multi-agent workflows for customer support"

PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the suite off the network even when a developer .env holds real keys.

    Env vars outrank the .env file in pydantic-settings, so blanking them forces every
    client - including the ones the CLI builds internally - onto the offline backends.
    """

    for name in PROVIDER_ENV:
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
