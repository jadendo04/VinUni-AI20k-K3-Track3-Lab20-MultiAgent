"""Benchmark: single-agent vs multi-agent on the same query set.

Metrics measured here:

* latency - wall clock per query
* cost - summed from per-call token usage and a static price table
* quality - deterministic heuristic proxy (0-10); peer review overrides it in the report
* citation coverage - share of retrieved sources actually cited in the answer
* failure rate - runs that errored or degraded to the fallback answer
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from statistics import mean
from time import perf_counter

from multi_agent_research_lab.agents.critic import citation_coverage
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import FALLBACK_PREFIX

Runner = Callable[[str], ResearchState]

TARGET_ANSWER_CHARS = 800
STOPWORDS = {"the", "and", "for", "with", "a", "an", "of", "to", "in", "on", "write", "word"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if len(t) > 2 and t not in STOPWORDS}


def quality_score(state: ResearchState) -> float:
    """Heuristic answer quality in 0-10.

    Deliberately cheap and deterministic so it can run in CI. It rewards: producing a
    non-degraded answer, adequate length, citing the retrieved sources, going through an
    analysis pass, and covering the terms of the question.
    """

    answer = state.final_answer or ""
    if not answer or answer.startswith(FALLBACK_PREFIX):
        return 0.0

    score = 3.0
    score += 2.0 * min(len(answer) / TARGET_ANSWER_CHARS, 1.0)
    score += 3.0 * citation_coverage(answer, len(state.sources))
    score += 1.0 if state.analysis_notes else 0.0
    query_terms = _tokens(state.request.query)
    if query_terms:
        score += 1.0 * len(query_terms & _tokens(answer)) / len(query_terms)
    return round(min(score, 10.0), 2)


def is_failed(state: ResearchState) -> bool:
    """A run counts as failed when it produced no usable answer."""

    answer = state.final_answer or ""
    return not answer or answer.startswith(FALLBACK_PREFIX)


def _notes(state: ResearchState) -> str:
    routes = "->".join(state.route_history) or "n/a"
    backends = {
        str(r.metadata.get("llm_backend"))
        for r in state.agent_results
        if r.metadata.get("llm_backend")
    }
    backend = ",".join(sorted(backends)) or "n/a"
    errors = f"; errors: {len(state.errors)}" if state.errors else ""
    return f"routes: {routes}; llm: {backend}; calls: {state.llm_calls}{errors}"


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one query through ``runner`` and measure it."""

    started = perf_counter()
    try:
        state = runner(query)
        latency = perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - a crash is a data point, not a stop
        latency = perf_counter() - started
        raise RuntimeError(f"{run_name} crashed after {latency:.2f}s: {exc}") from exc

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.total_cost_usd,
        quality_score=quality_score(state),
        citation_coverage=citation_coverage(state.final_answer or "", len(state.sources)),
        failure_rate=1.0 if is_failed(state) else 0.0,
        notes=_notes(state),
    )
    return state, metrics


def run_suite(
    run_name: str, queries: Sequence[str], runner: Runner
) -> tuple[list[ResearchState], BenchmarkMetrics]:
    """Run a whole query set and return the states plus averaged metrics."""

    states: list[ResearchState] = []
    per_query: list[BenchmarkMetrics] = []
    for query in queries:
        state, metrics = run_benchmark(run_name, query, runner)
        states.append(state)
        per_query.append(metrics)

    aggregate = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=mean(m.latency_seconds for m in per_query),
        estimated_cost_usd=sum(m.estimated_cost_usd or 0.0 for m in per_query),
        quality_score=round(mean(m.quality_score or 0.0 for m in per_query), 2),
        citation_coverage=round(mean(m.citation_coverage or 0.0 for m in per_query), 3),
        failure_rate=round(mean(m.failure_rate or 0.0 for m in per_query), 3),
        notes=f"{len(queries)} queries; total tokens "
        f"{sum(s.total_input_tokens + s.total_output_tokens for s in states)}",
    )
    return states, aggregate
