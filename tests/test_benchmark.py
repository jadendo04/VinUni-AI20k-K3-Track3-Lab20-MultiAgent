"""Benchmark and report tests."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    is_failed,
    quality_score,
    run_benchmark,
    run_suite,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import FALLBACK_PREFIX, MultiAgentWorkflow


def _runner(settings, workers):
    def run(query: str) -> ResearchState:
        workflow = MultiAgentWorkflow(settings=settings, workers=workers, use_langgraph=False)
        return workflow.run(ResearchState(request=ResearchQuery(query=query)))

    return run


def test_run_benchmark_reports_all_metrics(settings, workers) -> None:
    state, metrics = run_benchmark(
        "multi-agent", "Summarize LLM guardrails", _runner(settings, workers)
    )
    assert metrics.latency_seconds > 0
    assert metrics.quality_score and metrics.quality_score > 0
    assert metrics.citation_coverage == 1.0
    assert metrics.failure_rate == 0.0
    assert "routes:" in metrics.notes
    assert state.final_answer


def test_run_suite_averages_across_queries(settings, workers) -> None:
    queries = ["Summarize LLM guardrails", "Compare RAG and GraphRAG"]
    states, aggregate = run_suite("multi-agent", queries, _runner(settings, workers))
    assert len(states) == 2
    assert aggregate.run_name == "multi-agent"
    assert "2 queries" in aggregate.notes


def test_degraded_answer_scores_zero_and_counts_as_failure() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.final_answer = f"{FALLBACK_PREFIX} nothing produced"
    assert quality_score(state) == 0.0
    assert is_failed(state)


def test_report_includes_per_run_detail(settings, workers) -> None:
    states, aggregate = run_suite(
        "multi-agent", ["Summarize LLM guardrails"], _runner(settings, workers)
    )
    report = render_markdown_report(
        [aggregate], runs={"multi-agent": states}, queries=["Summarize LLM guardrails"], notes="ok"
    )
    assert "Benchmark Report" in report
    assert "Per-run detail" in report
    assert "researcher -> analyst -> writer" in report


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report
