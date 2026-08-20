"""Benchmark report rendering."""

from __future__ import annotations

from datetime import UTC, datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import tracing_backend


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    runs: dict[str, list[ResearchState]] | None = None,
    queries: list[str] | None = None,
    notes: str = "",
) -> str:
    """Render benchmark metrics to markdown.

    ``runs`` maps a run name to the states it produced, which adds a per-query breakdown
    and a route/failure section on top of the summary table.
    """

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Benchmark Report",
        "",
        f"- Generated: {generated}",
        f"- Tracing backend: `{tracing_backend()}`",
    ]
    if queries:
        lines.append(f"- Queries: {len(queries)}")
    lines += [
        "",
        "## Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    if queries:
        lines += ["", "## Query set", ""]
        lines += [f"{i}. {q}" for i, q in enumerate(queries, start=1)]

    if runs:
        lines += ["", "## Per-run detail", ""]
        for run_name, states in runs.items():
            lines += [f"### {run_name}", ""]
            for state in states:
                answer = state.final_answer or ""
                lines += [
                    f"- **Query:** {state.request.query}",
                    f"  - Routes: `{' -> '.join(state.route_history) or 'n/a'}`",
                    f"  - LLM calls: {state.llm_calls}, "
                    f"tokens in/out: {state.total_input_tokens}/{state.total_output_tokens}",
                    f"  - Sources: {len(state.sources)}, answer chars: {len(answer)}",
                    f"  - Errors: {', '.join(state.errors) if state.errors else 'none'}",
                ]
            lines.append("")

    if notes:
        lines += ["## Analysis", "", notes, ""]
    return "\n".join(lines) + "\n"
