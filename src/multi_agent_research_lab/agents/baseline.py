"""Single-agent baseline: one agent does search, analysis, and writing in one call.

This is the control arm of the benchmark. It is intentionally simple - fewer LLM calls,
lower latency, no handoffs - so the report can show what the multi-agent split actually
buys and what it costs.
"""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import format_sources
from multi_agent_research_lab.agents.writer import CITATION_RE, append_reference_list
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client
from multi_agent_research_lab.services.search_client import SearchClient, get_search_client

SYSTEM_PROMPT = (
    "You are a single-agent research assistant. In one pass you must gather the relevant "
    "evidence from the supplied sources, weigh it, and write the final answer for the "
    "stated audience, keeping [n] citations on every substantive claim."
)


class SingleAgentBaseline(BaseAgent):
    """One agent, one LLM call, whole task."""

    name = "baseline"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self.llm = llm or get_llm_client(temperature=0.3)
        self.search = search or get_search_client()

    def run(self, state: ResearchState) -> ResearchState:
        request = state.request
        state.sources = self.search.search(request.query, max_results=request.max_sources)
        prompt = (
            f"Query: {request.query}\n"
            f"Audience: {request.audience}\n\n"
            f"Sources:\n{format_sources(state.sources)}\n\n"
            "Answer directly: a lead paragraph, then 3-5 bullets with [n] citations, then "
            "one line naming the main trade-off or open risk."
        )
        response = self.llm.complete(SYSTEM_PROMPT, prompt)
        answer = response.content.strip()
        if not answer:
            raise AgentExecutionError("baseline produced empty answer")

        answer = append_reference_list(answer, state)
        state.final_answer = answer
        state.record_route("baseline")
        state.record_result(
            self._result(
                AgentName.WRITER,
                response,
                {
                    "citations": len(set(CITATION_RE.findall(answer))),
                    "source_count": len(state.sources),
                    "llm_backend": self.llm.backend,
                    "search_backend": self.search.backend,
                    "mode": "single-agent",
                },
            )
        )
        return state
