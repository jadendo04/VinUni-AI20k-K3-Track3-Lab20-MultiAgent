"""Critic agent: cheap, deterministic validation of the writer's output."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

CITATION_RE = re.compile(r"\[(\d+)\]")
MIN_ANSWER_CHARS = 200


def citation_coverage(answer: str, source_count: int) -> float:
    """Share of available sources actually cited in the answer (0-1)."""

    if source_count <= 0:
        return 0.0
    cited = {int(n) for n in CITATION_RE.findall(answer) if 1 <= int(n) <= source_count}
    return len(cited) / source_count


class CriticAgent(BaseAgent):
    """Fact-check and safety-review pass over the final answer."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate the final answer and append findings to the state."""

        answer = state.final_answer
        if not answer:
            raise AgentExecutionError("critic requires a final_answer")

        issues: list[str] = []
        coverage = citation_coverage(answer, len(state.sources))
        if coverage == 0.0 and state.sources:
            issues.append("no valid citation markers found")
        if len(answer) < MIN_ANSWER_CHARS:
            issues.append(f"answer shorter than {MIN_ANSWER_CHARS} characters")

        invalid = {
            int(n) for n in CITATION_RE.findall(answer) if not 1 <= int(n) <= len(state.sources)
        }
        if invalid:
            issues.append(f"citations point to non-existent sources: {sorted(invalid)}")
        if not state.analysis_notes:
            issues.append("answer written without an analysis pass")

        verdict = "pass" if not issues else "needs-revision"
        state.record_result(
            AgentResult(
                agent=AgentName.CRITIC,
                content=verdict if not issues else "; ".join(issues),
                metadata={
                    "verdict": verdict,
                    "citation_coverage": round(coverage, 3),
                    "issues": issues,
                    "llm_calls": 0,
                },
            )
        )
        for issue in issues:
            state.errors.append(f"critic: {issue}")
        return state
