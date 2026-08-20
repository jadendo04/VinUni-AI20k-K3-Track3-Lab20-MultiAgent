"""Supervisor / router.

The routing policy is deterministic on purpose: the decision is fully derivable from the
shared state ("do we have sources? analysis? an answer?"), so it is cheap, reproducible,
and unit-testable. An LLM router is only worth its latency and cost when the dispatch is
genuinely open-ended.
"""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(
        self,
        settings: Settings | None = None,
        max_iterations: int | None = None,
        max_agent_retries: int = 2,
        enable_critic: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_iterations = max_iterations or self.settings.max_iterations
        self.max_agent_retries = max_agent_retries
        self.enable_critic = enable_critic

    def decide(self, state: ResearchState) -> str:
        """Return the next route: researcher, analyst, writer, critic, or done."""

        # Guardrail 1: hard iteration cap - never loop forever.
        if state.iteration >= self.max_iterations:
            return DONE

        if state.final_answer:
            if self.enable_critic and "critic" not in state.route_history:
                return "critic"
            return DONE

        # Guardrail 2: an agent that keeps failing is skipped, not retried forever.
        if not state.research_notes and self._may_run(state, "researcher"):
            return "researcher"
        if state.research_notes and not state.analysis_notes and self._may_run(state, "analyst"):
            return "analyst"
        if (state.research_notes or state.analysis_notes) and self._may_run(state, "writer"):
            return "writer"
        return DONE

    def _may_run(self, state: ResearchState, agent: str) -> bool:
        return state.failure_count(agent) < self.max_agent_retries

    def run(self, state: ResearchState) -> ResearchState:
        """Record the next route in ``state.route_history`` / ``state.next_route``."""

        route = self.decide(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor.route",
            {
                "next": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
                "failures": dict(state.failed_agents),
            },
        )
        return state
