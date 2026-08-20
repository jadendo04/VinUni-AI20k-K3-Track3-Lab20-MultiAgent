"""Writer agent."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import format_sources
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client

SYSTEM_PROMPT = (
    "You are the Writer agent in a multi-agent research system. "
    "You do not search and you do not re-analyse; you synthesise the notes you are given "
    "into the final answer for the stated audience. Every substantive claim keeps its [n] "
    "citation. Do not invent sources or citation numbers."
)

CITATION_RE = re.compile(r"\[(\d+)\]")


def append_reference_list(answer: str, state: ResearchState) -> str:
    """Append a numbered reference list matching the [n] markers used in the answer."""

    if not state.sources or "## Sources" in answer:
        return answer
    references = "\n".join(
        f"[{i}] {doc.title} - {doc.url or 'no url'}" for i, doc in enumerate(state.sources, start=1)
    )
    return f"{answer}\n\n## Sources\n{references}"


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm_client(temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer``."""

        if not state.research_notes and not state.analysis_notes:
            raise AgentExecutionError("writer requires research or analysis notes")

        prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes or 'none'}\n\n"
            f"Analysis notes:\n{state.analysis_notes or 'none'}\n\n"
            f"Sources:\n{format_sources(state.sources)}\n\n"
            "Write the final answer: a short lead paragraph, then 3-5 bullets with [n] "
            "citations, then one line naming the main trade-off or open risk."
        )
        response = self.llm.complete(SYSTEM_PROMPT, prompt)
        answer = response.content.strip()
        if not answer:
            raise AgentExecutionError("writer produced empty answer")

        answer = append_reference_list(answer, state)
        state.final_answer = answer
        state.record_result(
            self._result(
                AgentName.WRITER,
                response,
                {
                    "citations": len(set(CITATION_RE.findall(answer))),
                    "llm_backend": self.llm.backend,
                },
            )
        )
        return state
