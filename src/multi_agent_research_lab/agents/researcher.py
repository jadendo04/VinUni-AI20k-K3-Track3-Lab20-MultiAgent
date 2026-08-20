"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client
from multi_agent_research_lab.services.search_client import SearchClient, get_search_client

SYSTEM_PROMPT = (
    "You are the Researcher agent in a multi-agent research system. "
    "You only gather and condense evidence; you never write the final answer. "
    "Cite every claim with the source index it came from, e.g. [1]. "
    "If the evidence is thin, say so explicitly instead of guessing."
)


def format_sources(sources: list[SourceDocument]) -> str:
    """Render sources as a numbered block the LLM can cite by index."""

    return "\n".join(
        f"[{i}] {doc.title} ({doc.url or 'no url'}): {doc.snippet}"
        for i, doc in enumerate(sources, start=1)
    )


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self.llm = llm or get_llm_client(temperature=0.2)
        self.search = search or get_search_client()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""

        request = state.request
        sources = self.search.search(request.query, max_results=request.max_sources)
        if not sources:
            raise AgentExecutionError("search returned no sources")

        state.sources = sources
        prompt = (
            f"Query: {request.query}\n"
            f"Audience: {request.audience}\n\n"
            f"Sources:\n{format_sources(sources)}\n\n"
            "Write research notes: 4-6 bullets of factual findings with [n] citations, "
            "then one line 'Open question:' naming what the sources do not cover."
        )
        response = self.llm.complete(SYSTEM_PROMPT, prompt)
        if not response.content.strip():
            raise AgentExecutionError("researcher produced empty notes")

        state.research_notes = response.content.strip()
        state.record_result(
            self._result(
                AgentName.RESEARCHER,
                response,
                {
                    "source_count": len(sources),
                    "search_backend": self.search.backend,
                    "llm_backend": self.llm.backend,
                },
            )
        )
        return state
