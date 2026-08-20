"""Command-line entrypoint for the lab."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from multi_agent_research_lab.agents.baseline import SingleAgentBaseline
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_suite
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    export_trace,
    summarize_trace,
    tracing_backend,
)
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

DEFAULT_CONFIG = Path("configs/lab_default.yaml")


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _print_state(state: ResearchState, title: str, as_json: bool) -> None:
    if as_json:
        console.print_json(state.model_dump_json())
        return

    console.print(Panel(Text(state.final_answer or "(no answer)"), title=title))

    table = Table(title="Run summary", show_header=True)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("routes", " -> ".join(state.route_history) or "n/a")
    table.add_row("iterations", str(state.iteration))
    table.add_row("sources", str(len(state.sources)))
    table.add_row("llm calls", str(state.llm_calls))
    table.add_row("tokens in/out", f"{state.total_input_tokens}/{state.total_output_tokens}")
    table.add_row("est. cost (USD)", f"{state.total_cost_usd:.6f}")
    table.add_row("tracing backend", tracing_backend())
    table.add_row("errors", ", ".join(state.errors) or "none")
    console.print(table)

    if state.trace:
        console.print(Panel(Text(summarize_trace(state.trace)), title="Trace"))


def _run_baseline(query: str) -> ResearchState:
    state = ResearchState(request=_parse_query(query))
    return SingleAgentBaseline().execute(state)


def _run_multi(query: str, enable_critic: bool = True, use_langgraph: bool = True) -> ResearchState:
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow(enable_critic=enable_critic, use_langgraph=use_langgraph)
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    as_json: Annotated[bool, typer.Option("--json", help="Print raw state as JSON")] = False,
) -> None:
    """Run the single-agent baseline: one agent does search, analysis, and writing."""

    _init()
    state = _run_baseline(query)
    _print_state(state, "Single-Agent Baseline", as_json)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic/--no-critic", help="Run critic pass")] = True,
    langgraph: Annotated[
        bool, typer.Option("--langgraph/--no-langgraph", help="Execution engine")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print raw state as JSON")] = False,
    trace_out: Annotated[
        Path | None, typer.Option("--trace-out", help="Write the run trace to JSON")
    ] = None,
) -> None:
    """Run the multi-agent workflow: Supervisor -> Researcher -> Analyst -> Writer."""

    _init()
    state = _run_multi(query, enable_critic=critic, use_langgraph=langgraph)
    _print_state(state, "Multi-Agent Workflow", as_json)
    if trace_out is not None:
        path = export_trace(state.trace, trace_out)
        console.print(f"trace written to {path}")


@app.command()
def benchmark(
    config: Annotated[
        Path, typer.Option("--config", help="YAML config holding benchmark queries")
    ] = DEFAULT_CONFIG,
    query: Annotated[
        list[str] | None, typer.Option("--query", "-q", help="Override query set (repeatable)")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Report path")] = Path(
        "reports/benchmark_report.md"
    ),
    critic: Annotated[bool, typer.Option("--critic/--no-critic")] = True,
    notes_file: Annotated[
        Path,
        typer.Option("--notes-file", help="Markdown appended to the report's Analysis section"),
    ] = Path("docs/benchmark_notes.md"),
) -> None:
    """Run both pipelines over the same queries and write a markdown report."""

    _init()
    queries = list(query) if query else _load_queries(config)
    if not queries:
        console.print(Panel.fit("No benchmark queries found", title="Input Error", style="red"))
        raise typer.Exit(code=1)

    console.print(f"Running {len(queries)} queries through both pipelines...")
    baseline_states, baseline_metrics = run_suite("single-agent", queries, _run_baseline)
    multi_states, multi_metrics = run_suite(
        "multi-agent", queries, lambda q: _run_multi(q, enable_critic=critic)
    )

    report = render_markdown_report(
        [baseline_metrics, multi_metrics],
        runs={"single-agent": baseline_states, "multi-agent": multi_states},
        queries=queries,
        notes=_analysis(baseline_metrics, multi_metrics, notes_file),
    )
    store = LocalArtifactStore(root=out.parent if out.parent.name else Path("reports"))
    path = store.write_text(out.name, report)
    console.print(Panel(Text(report), title=str(path)))


def _load_queries(config: Path) -> list[str]:
    if not config.exists():
        return []
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return list(data.get("benchmark", {}).get("queries", []))


def _analysis(base: BenchmarkMetrics, multi: BenchmarkMetrics, notes_file: Path) -> str:
    """Generated numbers first, then the hand-written analysis kept under version control."""

    parts = [_auto_analysis(base, multi)]
    if notes_file.exists():
        parts.append(notes_file.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def _auto_analysis(base: BenchmarkMetrics, multi: BenchmarkMetrics) -> str:
    """Small generated paragraph; the hand-written analysis lives in the report file."""

    latency_ratio = multi.latency_seconds / max(base.latency_seconds, 1e-9)
    cost_delta = (multi.estimated_cost_usd or 0.0) - (base.estimated_cost_usd or 0.0)
    quality_delta = (multi.quality_score or 0.0) - (base.quality_score or 0.0)
    return (
        f"Multi-agent took {latency_ratio:.2f}x the baseline latency, "
        f"cost {cost_delta:+.4f} USD more across the query set, and scored "
        f"{quality_delta:+.2f} on the heuristic quality proxy. "
        "Quality here is a heuristic (length, citation coverage, analysis pass, term "
        "coverage), not a human rubric - use peer review for the final judgement."
    )


if __name__ == "__main__":
    app()
