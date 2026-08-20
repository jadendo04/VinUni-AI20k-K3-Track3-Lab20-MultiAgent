"""LLM client abstraction.

Production note: agents depend on this interface instead of importing a provider SDK
directly. Retry, timeout, token accounting, and cost estimation live here so agents stay
focused on prompting.

Two backends are supported:

* ``openai`` — used automatically when ``OPENAI_API_KEY`` is set and the ``openai``
  package is installed.
* ``offline`` — a deterministic, dependency-free stub used when no key is configured.
  It keeps the whole workflow runnable (and unit-testable) without network access. Its
  output is prefixed with ``[offline-stub]`` so it can never be mistaken for a real
  model answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)
_LOGGED: set[str] = set()


def _log_once(message: str) -> None:
    """Log a backend-selection notice once per process instead of per client."""

    if message not in _LOGGED:
        _LOGGED.add(message)
        logger.info(message)


OFFLINE_PREFIX = "[offline-stub]"

# USD per 1M tokens. Extend when adding models.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate request cost from a static price table."""

    price_in, price_out = PRICING_USD_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) used by the offline backend."""

    return max(1, len(text) // 4)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        force_offline: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.openai_model
        self.temperature = temperature
        self._client: Any = None if force_offline else self._build_openai_client()
        self.backend = "openai" if self._client is not None else "offline"

    def _build_openai_client(self) -> Any | None:
        if not self.settings.openai_api_key:
            _log_once("OPENAI_API_KEY not set - using deterministic offline LLM backend")
            return None
        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover - depends on optional extra
            logger.warning("openai package missing - using offline LLM backend")
            return None
        return OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=float(self.settings.timeout_seconds),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion, retrying transient provider failures."""

        if self._client is None:
            return self._complete_offline(system_prompt, user_prompt)
        try:
            return self._complete_openai(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - surface one domain error to callers
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        cost = estimate_cost_usd(self.model, input_tokens or 0, output_tokens or 0)
        logger.debug(
            "llm call model=%s in=%s out=%s cost=%.6f",
            self.model,
            input_tokens,
            output_tokens,
            cost,
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _complete_offline(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        content = _offline_completion(system_prompt, user_prompt)
        input_tokens = approx_tokens(system_prompt + user_prompt)
        output_tokens = approx_tokens(content)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
        )


def _detect_role(system_prompt: str) -> str:
    lowered = system_prompt.lower()
    for role in ("researcher", "analyst", "writer", "critic", "supervisor", "single-agent"):
        if role in lowered:
            return role
    return "assistant"


def _extract_query(user_prompt: str) -> str:
    match = re.search(r"(?im)^\s*(?:query|question|task)\s*:\s*(.+)$", user_prompt)
    return match.group(1).strip() if match else user_prompt.strip().splitlines()[0][:160]


# Prompt lines that are instructions to the model, not context to be summarised.
_INSTRUCTION_PREFIXES = (
    "query:",
    "audience:",
    "sources:",
    "research notes:",
    "analysis notes:",
    "write ",
    "produce:",
    "answer directly",
    "then ",
)


def _extract_context_lines(user_prompt: str, limit: int = 6) -> list[str]:
    """Pick the context lines of a prompt, dropping the instruction lines."""

    lines = [line.strip(" -*\t") for line in user_prompt.splitlines()]
    picked = [
        line
        for line in lines
        if len(line) > 30 and not line.lower().startswith(_INSTRUCTION_PREFIXES)
    ]
    return picked[:limit]


def _offline_completion(system_prompt: str, user_prompt: str) -> str:
    """Deterministic stand-in for a real completion.

    It re-uses the context that was put into the prompt, so the workflow still exercises
    real handoffs (each agent must pass usable context downstream) without a provider.
    """

    role = _detect_role(system_prompt)
    query = _extract_query(user_prompt)
    context = _extract_context_lines(user_prompt)
    bullets = "\n".join(f"- {line}" for line in context) or "- No upstream context provided."

    if role == "researcher":
        body = f"Research notes for: {query}\n{bullets}\nOpen question: coverage of recent work."
    elif role == "analyst":
        body = (
            f"Analysis for: {query}\nKey claims:\n{bullets}\n"
            "Agreement: sources broadly align on the core definition.\n"
            "Weak evidence: vendor material and undated pages need corroboration."
        )
    elif role == "writer":
        body = (
            f"{query}\n\nSummary\n{bullets}\n\n"
            "Trade-offs: added coordination cost buys role isolation and traceability."
        )
    elif role == "critic":
        body = f"Review of the draft for: {query}\n{bullets}\nVerdict: acceptable with citations."
    else:
        body = f"Answer for: {query}\n{bullets}"

    return f"{OFFLINE_PREFIX} {body}"


def get_llm_client(temperature: float = 0.2, model: str | None = None) -> LLMClient:
    """Factory used by agents so backend selection stays in one place."""

    return LLMClient(temperature=temperature, model=model)
