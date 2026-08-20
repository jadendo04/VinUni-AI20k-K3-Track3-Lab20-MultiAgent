"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client

SYSTEM_PROMPT = (
    "You are the Analyst agent in a multi-agent research system. "
    "You never search and you never write the final prose. "
    "You compare the research notes, extract the key claims, mark where sources agree or "
    "disagree, and flag weak evidence (vendor content, undated pages, single-source claims)."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm_client(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``."""

        if not state.research_notes:
            raise AgentExecutionError("analyst requires research_notes")

        prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Produce: 'Key claims:' (3-5 bullets with [n] citations), "
            "'Agreement:' one line, 'Disagreement:' one line, "
            "'Weak evidence:' one line naming which sources need corroboration."
        )
        response = self.llm.complete(SYSTEM_PROMPT, prompt)
        if not response.content.strip():
            raise AgentExecutionError("analyst produced empty analysis")

        state.analysis_notes = response.content.strip()
        state.record_result(
            self._result(AgentName.ANALYST, response, {"llm_backend": self.llm.backend})
        )
        return state
