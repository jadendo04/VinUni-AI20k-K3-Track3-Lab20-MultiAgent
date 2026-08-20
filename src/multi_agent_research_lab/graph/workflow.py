"""Multi-agent workflow orchestration.

The graph is Supervisor-centric: every worker returns control to the supervisor, which
re-reads the shared state and picks the next route or stops. Orchestration lives here;
agent internals stay in ``agents/``.

Two execution engines are supported and produce the same result:

* LangGraph (default when the ``llm`` extra is installed) - real nodes, conditional
  edges, and a recursion limit derived from ``max_iterations``.
* a built-in loop - used when LangGraph is unavailable, and handy in tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.utils.timer import elapsed_timer

logger = logging.getLogger(__name__)

FALLBACK_PREFIX = "[degraded]"


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        workers: dict[str, BaseAgent] | None = None,
        enable_critic: bool = False,
        use_langgraph: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.enable_critic = enable_critic
        self.use_langgraph = use_langgraph
        self.supervisor = supervisor or SupervisorAgent(
            settings=self.settings, enable_critic=enable_critic
        )
        self.workers: dict[str, BaseAgent] = workers or {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
            "critic": CriticAgent(),
        }

    # ------------------------------------------------------------------ graph
    def build(self) -> object:
        """Create and compile the LangGraph graph.

        Nodes: supervisor + one node per worker. Every worker edges back to the
        supervisor; the supervisor's conditional edge dispatches on ``state.next_route``
        and terminates on ``done``.
        """

        from langgraph.graph import END, StateGraph

        graph: Any = StateGraph(ResearchState)
        graph.add_node("supervisor", self._node(self.supervisor))
        for name, agent in self.workers.items():
            graph.add_node(name, self._node(agent))
            graph.add_edge(name, "supervisor")

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _next_route,
            {**{name: name for name in self.workers}, DONE: END},
        )
        return graph.compile()

    def _node(self, agent: BaseAgent) -> Callable[[ResearchState], dict[str, object]]:
        """Wrap an agent as a graph node returning a state update dict."""

        def node(state: ResearchState) -> dict[str, object]:
            return agent.execute(state).model_dump()

        return node

    # -------------------------------------------------------------- execution
    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return the final state."""

        with trace_span("workflow.multi_agent", {"query": state.request.query}) as span:
            with elapsed_timer() as elapsed:
                if self.use_langgraph:
                    try:
                        state = self._run_langgraph(state)
                    except ImportError:
                        logger.warning("langgraph not installed - falling back to built-in loop")
                        state = self._run_loop(state)
                else:
                    state = self._run_loop(state)
                duration = elapsed()
            span["attributes"]["route_history"] = list(state.route_history)

        state = self._finalize(state)
        state.add_trace_event(
            "workflow.completed",
            {
                "duration_seconds": duration,
                "detail": f"routes={'->'.join(state.route_history)}",
                "iterations": state.iteration,
                "errors": list(state.errors),
            },
        )
        return state

    def _run_langgraph(self, state: ResearchState) -> ResearchState:
        app: Any = self.build()
        # Recursion limit is the second guardrail behind the supervisor's iteration cap.
        result = app.invoke(state, config={"recursion_limit": self.settings.max_iterations * 2 + 4})
        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)

    def _run_loop(self, state: ResearchState) -> ResearchState:
        """Engine-free execution of the same policy: supervisor -> worker -> supervisor."""

        while True:
            state = self.supervisor.execute(state)
            route = state.next_route or DONE
            if route == DONE:
                return state
            agent = self.workers.get(route)
            if agent is None:
                state.record_failure("supervisor", f"unknown route: {route}")
                return state
            state = agent.execute(state)

    def _finalize(self, state: ResearchState) -> ResearchState:
        """Guarantee an answer even when the workflow degraded."""

        if state.final_answer:
            return state
        partial = state.analysis_notes or state.research_notes
        reason = "; ".join(state.errors) or "stopped by max_iterations guardrail"
        if partial:
            state.final_answer = (
                f"{FALLBACK_PREFIX} Writer did not complete ({reason}). "
                f"Best available material:\n\n{partial}"
            )
        else:
            state.final_answer = (
                f"{FALLBACK_PREFIX} No answer produced ({reason}). "
                "Check search/LLM backends and the trace for the failing step."
            )
        return state


def _next_route(state: ResearchState | dict[str, Any]) -> str:
    """Conditional-edge selector reading the supervisor's decision from the state."""

    route = state.next_route if isinstance(state, ResearchState) else state.get("next_route")
    return str(route or DONE)
