"""Search client abstraction for ResearcherAgent.

Backends:

* ``tavily`` — used when ``TAVILY_API_KEY`` is set (plain ``urllib`` call, no extra
  dependency). SSL verification uses ``certifi`` when available, which is the usual fix
  for ``SSLCertVerificationError`` on macOS (see ``docs/lab_guide.md``).
* ``mock`` — a small local corpus, so the lab runs offline and tests stay deterministic.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)
_LOGGED: set[str] = set()


def _log_once(message: str) -> None:
    """Log a backend-selection notice once per process instead of per client."""

    if message not in _LOGGED:
        _LOGGED.add(message)
        logger.info(message)


TAVILY_ENDPOINT = "https://api.tavily.com/search"

MOCK_CORPUS: list[dict[str, str]] = [
    {
        "title": "Anthropic - Building effective agents",
        "url": "https://www.anthropic.com/engineering/building-effective-agents",
        "snippet": (
            "Start with the simplest pattern that works: a single well-prompted LLM call. "
            "Add orchestration only when a task decomposes into parallel or specialised "
            "subtasks, because every extra agent adds latency, cost, and failure surface."
        ),
        "keywords": "agent multi-agent orchestration workflow guardrail design",
    },
    {
        "title": "LangGraph concepts - graphs, state, and conditional edges",
        "url": "https://langchain-ai.github.io/langgraph/concepts/",
        "snippet": (
            "LangGraph models an agent system as a graph: nodes mutate a shared typed "
            "state, conditional edges route control flow, and recursion limits stop runaway "
            "loops between a supervisor and its workers."
        ),
        "keywords": "langgraph graph state routing supervisor workflow loop",
    },
    {
        "title": "GraphRAG: knowledge-graph grounded retrieval",
        "url": "https://arxiv.org/abs/2404.16130",
        "snippet": (
            "GraphRAG builds an entity graph from a corpus and answers global questions by "
            "summarising communities in that graph, outperforming plain vector RAG on "
            "sense-making queries at a higher indexing cost."
        ),
        "keywords": "graphrag rag retrieval knowledge graph summary state-of-the-art",
    },
    {
        "title": "OpenAI Agents SDK - orchestration and handoffs",
        "url": "https://developers.openai.com/api/docs/guides/agents/orchestration",
        "snippet": (
            "Handoffs let one agent transfer a conversation plus context to a specialised "
            "agent. Deterministic routing is preferable when the decision can be expressed "
            "as a rule; model-based routing is for genuinely open-ended dispatch."
        ),
        "keywords": "handoff routing orchestration agents sdk supervisor customer support",
    },
    {
        "title": "Production guardrails for LLM agents",
        "url": "https://docs.smith.langchain.com/",
        "snippet": (
            "Guardrails that matter in production: hard iteration caps, per-call timeouts, "
            "bounded retries with backoff, output schema validation, and a trace for every "
            "run so failures can be attributed to a specific step."
        ),
        "keywords": "guardrail timeout retry validation trace observability production llm",
    },
    {
        "title": "Benchmarking single-agent vs multi-agent pipelines",
        "url": "https://langfuse.com/docs",
        "snippet": (
            "Compare pipelines on the same query set with latency, token cost, rubric "
            "quality, and citation coverage. Multi-agent typically wins on coverage and "
            "traceability, and loses on latency and cost."
        ),
        "keywords": "benchmark latency cost quality citation coverage single multi agent",
    },
]


class SearchClient:
    """Provider-agnostic search client."""

    def __init__(self, settings: Settings | None = None, force_mock: bool = False) -> None:
        self.settings = settings or get_settings()
        use_tavily = bool(self.settings.tavily_api_key) and not force_mock
        self.backend = "tavily" if use_tavily else "mock"
        if self.backend == "mock":
            _log_once("TAVILY_API_KEY not set - using local mock search corpus")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return documents relevant to ``query``, ranked best-first."""

        if self.backend == "tavily":
            try:
                return self._search_tavily(query, max_results)
            except AgentExecutionError as exc:
                logger.warning("tavily search failed (%s) - falling back to mock corpus", exc)
        return self._search_mock(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            TAVILY_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout_seconds, context=_ssl_context()
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise AgentExecutionError(f"tavily request failed: {exc}") from exc

        return [
            SourceDocument(
                title=item.get("title") or "Untitled",
                url=item.get("url"),
                snippet=(item.get("content") or "")[:800],
                metadata={"score": item.get("score"), "provider": "tavily"},
            )
            for item in data.get("results", [])[:max_results]
        ]

    def _search_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        terms = {t for t in _tokenize(query) if len(t) > 2}
        scored: list[tuple[float, dict[str, str]]] = []
        for doc in MOCK_CORPUS:
            haystack = set(_tokenize(f"{doc['title']} {doc['snippet']} {doc['keywords']}"))
            overlap = len(terms & haystack)
            scored.append((overlap / max(len(terms), 1), doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SourceDocument(
                title=doc["title"],
                url=doc["url"],
                snippet=doc["snippet"],
                metadata={"score": round(score, 3), "provider": "mock"},
            )
            for score, doc in scored[:max_results]
        ]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi is optional
        return ssl.create_default_context()


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(c.lower() if c.isalnum() else " " for c in text).split()]


def get_search_client() -> SearchClient:
    """Factory used by agents so backend selection stays in one place."""

    return SearchClient()
