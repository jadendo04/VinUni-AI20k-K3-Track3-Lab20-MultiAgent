# Benchmark Report

- Generated: 2026-08-20 16:06 UTC
- Tracing backend: `local`
- Queries: 3

## Summary

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent | 0.00 | 0.0000 | 9.0 | 100% | 0% | 3 queries; total tokens 2758 |
| multi-agent | 0.07 | 0.0000 | 10.0 | 100% | 0% | 3 queries; total tokens 11589 |

## Query set

1. Research GraphRAG state-of-the-art and write a 500-word summary
2. Compare single-agent and multi-agent workflows for customer support
3. Summarize production guardrails for LLM agents

## Per-run detail

### single-agent

- **Query:** Research GraphRAG state-of-the-art and write a 500-word summary
  - Routes: `baseline`
  - LLM calls: 1, tokens in/out: 505/414
  - Sources: 5, answer chars: 2178
  - Errors: none
- **Query:** Compare single-agent and multi-agent workflows for customer support
  - Routes: `baseline`
  - LLM calls: 1, tokens in/out: 507/416
  - Sources: 5, answer chars: 2179
  - Errors: none
- **Query:** Summarize production guardrails for LLM agents
  - Routes: `baseline`
  - LLM calls: 1, tokens in/out: 504/412
  - Sources: 5, answer chars: 2166
  - Errors: none

### multi-agent

- **Query:** Research GraphRAG state-of-the-art and write a 500-word summary
  - Routes: `researcher -> analyst -> writer -> critic -> done`
  - LLM calls: 3, tokens in/out: 2509/1357
  - Sources: 5, answer chars: 2351
  - Errors: none
- **Query:** Compare single-agent and multi-agent workflows for customer support
  - Routes: `researcher -> analyst -> writer -> critic -> done`
  - LLM calls: 3, tokens in/out: 2521/1365
  - Sources: 5, answer chars: 2356
  - Errors: none
- **Query:** Summarize production guardrails for LLM agents
  - Routes: `researcher -> analyst -> writer -> critic -> done`
  - LLM calls: 3, tokens in/out: 2494/1343
  - Sources: 5, answer chars: 2322
  - Errors: none

## Analysis

Multi-agent took 315.54x the baseline latency, cost +0.0000 USD more across the query set, and scored +1.00 on the heuristic quality proxy. Quality here is a heuristic (length, citation coverage, analysis pass, term coverage), not a human rubric - use peer review for the final judgement.

## Reading these numbers

The committed run used the **offline backends** (`OPENAI_API_KEY` and `TAVILY_API_KEY`
unset), so:

- **Cost is 0.00 USD** for both arms. Token counts are real (estimated at ~4 chars per
  token) but no provider was billed. With `gpt-4o-mini` pricing, the token volume above
  works out to roughly 3x baseline cost for the multi-agent arm — three LLM calls instead
  of one.
- **Latency ratio is not representative.** Offline completions return in microseconds, so
  what the multi-agent arm actually measures is graph compilation plus orchestration
  overhead. With a real provider, latency is dominated by the 3 sequential LLM calls, so
  expect ~2.5-3x the baseline, not the ratio printed above.
- **Quality is a heuristic proxy** (length, citation coverage, analysis pass, query-term
  coverage), not a human judgement. Use `docs/peer_review_rubric.md` for the real score.

Re-run with keys in `.env` to get billable numbers: `python -m multi_agent_research_lab.cli benchmark`.

## What multi-agent bought, and what it cost

| Dimension | Single-agent | Multi-agent |
|---|---|---|
| LLM calls per query | 1 | 3 (+1 free deterministic critic pass) |
| Failure surface | one call fails, the run fails | any worker can fail and the run still returns a degraded answer |
| Debuggability | one opaque output | `route_history` + per-step trace attributes the failure to one agent |
| Citation discipline | prompt-only | prompt + writer post-processing + critic validation |
| Latency / cost | best | ~3x |

The honest reading: on these three broad research queries the split mainly buys
**traceability and citation discipline**, not raw answer quality. That matches the
guidance in the sources - add agents when the task genuinely decomposes, not by default.

## Failure modes hit while building, and the fix

1. **Supervisor <-> worker loop.** If a worker returns without filling its state field,
   the supervisor routes to it again forever.
   *Fix:* three layers - `Settings.max_iterations` (hard cap in `SupervisorAgent.decide`),
   a per-agent failure counter (`max_agent_retries=2`, then the agent is skipped), and a
   LangGraph `recursion_limit` as the backstop.

2. **A crashing agent killed the whole run.** An exception in the researcher propagated
   out of the graph and the user got a stack trace instead of an answer.
   *Fix:* `BaseAgent.execute` catches, records `state.record_failure(...)`, and lets the
   supervisor re-route. `MultiAgentWorkflow._finalize` guarantees a `[degraded]` answer
   built from whatever notes exist, so the pipeline never returns nothing.

3. **Citations drifting from the retrieved sources.** The writer can cite `[7]` when only
   5 sources were retrieved, or drop citations entirely.
   *Fix:* sources are passed as a numbered block, the writer appends a deterministic
   `## Sources` list, and `CriticAgent` flags out-of-range markers and zero coverage.

4. **No API key would have blocked the whole lab.** Every agent hard-failed with a
   provider error.
   *Fix:* `LLMClient` / `SearchClient` fall back to deterministic offline backends and
   label output with `[offline-stub]`, so CI and the tests run without secrets and no
   stub output can be mistaken for a model answer.

5. **Terminal rendering hid the citations.** Rich interpreted `[1]` and `[offline-stub]`
   as markup and silently dropped them - the answer looked uncited when it was not.
   *Fix:* answers are printed via `rich.text.Text`, markup disabled.

