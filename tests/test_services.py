"""Service backend tests."""

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import (
    OFFLINE_PREFIX,
    LLMClient,
    estimate_cost_usd,
)
from multi_agent_research_lab.services.search_client import SearchClient


def test_offline_llm_is_deterministic_and_labelled(llm: LLMClient) -> None:
    first = llm.complete("You are the Writer agent.", "Query: what is GraphRAG?")
    second = llm.complete("You are the Writer agent.", "Query: what is GraphRAG?")
    assert llm.backend == "offline"
    assert first.content == second.content
    assert first.content.startswith(OFFLINE_PREFIX)
    assert first.input_tokens and first.output_tokens


def test_cost_estimate_uses_price_table() -> None:
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15
    assert estimate_cost_usd("unknown-model", 1_000_000, 1_000_000) == 0.0


def test_mock_search_ranks_by_term_overlap(search: SearchClient) -> None:
    results = search.search("guardrails timeout retry validation", max_results=3)
    assert search.backend == "mock"
    assert len(results) == 3
    assert "guardrail" in results[0].title.lower()


def test_openai_backend_selected_when_key_present() -> None:
    pytest.importorskip("openai")  # the openai SDK ships in the optional "llm" extra
    client = LLMClient(settings=Settings(OPENAI_API_KEY="sk-test"))
    assert client.backend == "openai"


def test_openrouter_backend_selected_from_base_url() -> None:
    pytest.importorskip("openai")
    client = LLMClient(
        settings=Settings(
            OPENAI_API_KEY="sk-or-test",
            OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            OPENAI_MODEL="openai/gpt-4o-mini",
        )
    )
    assert client.backend == "openrouter"
    assert estimate_cost_usd("openai/gpt-4o-mini", 1_000_000, 0) == 0.15
