"""CLI smoke tests (offline backends, no network)."""

from typer.testing import CliRunner

from multi_agent_research_lab.cli import app

runner = CliRunner()


def test_baseline_command_runs() -> None:
    result = runner.invoke(
        app, ["baseline", "-q", "Summarize production guardrails for LLM agents"]
    )
    assert result.exit_code == 0
    assert "Single-Agent Baseline" in result.stdout


def test_multi_agent_command_runs_and_exports_trace(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "multi-agent",
            "-q",
            "Summarize production guardrails for LLM agents",
            "--no-langgraph",
            "--trace-out",
            str(trace_path),
        ],
    )
    assert result.exit_code == 0
    assert trace_path.exists()


def test_short_query_is_rejected() -> None:
    result = runner.invoke(app, ["baseline", "-q", "hi"])
    assert result.exit_code == 1
