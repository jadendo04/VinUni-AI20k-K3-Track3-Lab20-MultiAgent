"""Shared state for the multi-agent workflow.

The state is the single source of truth handed from agent to agent. Every field is
either an input (``request``), an intermediate artifact (``sources``,
``research_notes``, ``analysis_notes``), an output (``final_answer``), or bookkeeping
used for guardrails and debugging (``iteration``, ``route_history``, ``trace``,
``errors``, ``failed_agents``).
"""

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)
    next_route: str | None = None

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    failed_agents: dict[str, int] = Field(default_factory=dict)

    def record_route(self, route: str) -> None:
        """Append a routing decision and advance the iteration counter."""

        self.route_history.append(route)
        self.next_route = route
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def record_result(self, result: AgentResult) -> None:
        """Store an agent output so downstream agents and the benchmark can read it."""

        self.agent_results.append(result)

    def record_failure(self, agent: str, message: str) -> None:
        """Track a per-agent failure so the supervisor can fall back instead of looping."""

        self.errors.append(f"{agent}: {message}")
        self.failed_agents[agent] = self.failed_agents.get(agent, 0) + 1

    def failure_count(self, agent: str) -> int:
        return self.failed_agents.get(agent, 0)

    @property
    def total_input_tokens(self) -> int:
        return sum(int(r.metadata.get("input_tokens") or 0) for r in self.agent_results)

    @property
    def total_output_tokens(self) -> int:
        return sum(int(r.metadata.get("output_tokens") or 0) for r in self.agent_results)

    @property
    def total_cost_usd(self) -> float:
        return sum(float(r.metadata.get("cost_usd") or 0.0) for r in self.agent_results)

    @property
    def llm_calls(self) -> int:
        return sum(int(r.metadata.get("llm_calls") or 0) for r in self.agent_results)
