"""Base agent contract.

Every agent reads the shared state, writes its own slice of it, and returns it. The
``execute`` wrapper adds the cross-cutting concerns the lab requires - a trace span per
step and a failure guard that converts an agent crash into recorded state instead of a
dead run, so the supervisor can fall back.
"""

from abc import ABC, abstractmethod
from typing import Any

from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMResponse


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""

    def execute(self, state: ResearchState) -> ResearchState:
        """Run the agent inside a trace span, capturing failures instead of raising."""

        with trace_span(f"agent.{self.name}", {"iteration": state.iteration}) as span:
            detail = "ok"
            try:
                state = self.run(state)
            except Exception as exc:  # noqa: BLE001 - guardrail: never kill the workflow
                state.record_failure(self.name, str(exc))
                span["error"] = str(exc)
                detail = f"failed: {exc}"
            state.add_trace_event(
                f"agent.{self.name}",
                {
                    "duration_seconds": span["duration_seconds"],
                    "iteration": state.iteration,
                    "detail": detail,
                },
            )
        return state

    def _result(
        self,
        agent: AgentName,
        response: LLMResponse,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Build an ``AgentResult`` carrying token/cost metadata for the benchmark."""

        payload: dict[str, Any] = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "llm_calls": 1,
        }
        payload.update(metadata or {})
        return AgentResult(agent=agent, content=response.content, metadata=payload)
