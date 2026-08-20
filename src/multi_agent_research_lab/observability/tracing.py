"""Tracing hooks.

This module deliberately does not bind to one provider. Every agent step opens a
``trace_span``; the span is always recorded in-process (and can be exported as JSON for
submission), and is additionally mirrored to LangSmith or Langfuse when the matching
keys are configured. Provider failures never break a run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


def tracing_backend() -> str:
    """Return the active tracing provider name: ``langsmith``, ``langfuse`` or ``local``."""

    settings = get_settings()
    if settings.langsmith_api_key:
        return "langsmith"
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        return "langfuse"
    return "local"


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Time a step and mirror it to the configured tracing provider."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    provider = _start_provider_span(name, attributes or {})
    try:
        yield span
    except Exception as exc:  # noqa: BLE001 - annotate the span, then re-raise unchanged
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        _end_provider_span(provider, span)


def _start_provider_span(name: str, attributes: dict[str, Any]) -> Any | None:
    backend = tracing_backend()
    settings = get_settings()
    try:
        if backend == "langsmith":
            from langsmith.run_helpers import trace as ls_trace

            manager = ls_trace(
                name=name, inputs=attributes, project_name=settings.langsmith_project
            )
            return (manager, manager.__enter__())
        if backend == "langfuse":
            from langfuse import Langfuse  # type: ignore[import-not-found]

            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            return (None, client.trace(name=name, input=attributes))
    except Exception as exc:  # noqa: BLE001 - tracing must never break the workflow
        logger.debug("tracing provider unavailable: %s", exc)
    return None


def _end_provider_span(provider: Any | None, span: dict[str, Any]) -> None:
    if provider is None:
        return
    manager, handle = provider
    try:
        if hasattr(handle, "end"):
            handle.end(outputs=span)
        if manager is not None:
            manager.__exit__(None, None, None)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the workflow
        logger.debug("failed to close provider span: %s", exc)


def export_trace(trace: list[dict[str, Any]], path: Path | str) -> Path:
    """Write a run trace to JSON - usable as submission evidence without a SaaS provider."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def summarize_trace(trace: list[dict[str, Any]]) -> str:
    """One-line-per-step summary for CLI output."""

    lines = []
    for event in trace:
        payload = event.get("payload", {})
        duration = payload.get("duration_seconds")
        suffix = f" ({duration:.2f}s)" if isinstance(duration, int | float) else ""
        detail = payload.get("detail", "")
        lines.append(f"{event.get('name')}{suffix} {detail}".rstrip())
    return "\n".join(lines)
