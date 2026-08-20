# Benchmark Report

- Generated: 2026-08-20 16:47 UTC
- Tracing backend: `local`
- Queries: 3

## Summary

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent | 5.95 | 0.0008 | 8.7 | 100% | 0% | 3 queries; total tokens 2374 |
| multi-agent | 14.66 | 0.0022 | 9.7 | 100% | 0% | 3 queries; total tokens 7638 |

## Query set

1. Research GraphRAG state-of-the-art and write a 500-word summary
2. Compare single-agent and multi-agent workflows for customer support
3. Summarize production guardrails for LLM agents

## Per-run detail

### single-agent

- **Query:** Research GraphRAG state-of-the-art and write a 500-word summary
  - Routes: `baseline`
  - LLM calls: 1, tokens in/out: 443/394
  - Sources: 5, answer chars: 2775
  - Errors: none
- **Query:** Compare single-agent and multi-agent workflows for customer support
  - Routes: `baseline`
  - LLM calls: 1, tokens in/out: 436/397
  - Sources: 5, answer chars: 2794
  - Errors: none
- **Query:** Summarize production guardrails for LLM agents
  - Routes: `baseline`
  - LLM calls: 1, tokens in/out: 444/260
  - Sources: 5, answer chars: 1930
  - Errors: none

### multi-agent

- **Query:** Research GraphRAG state-of-the-art and write a 500-word summary
  - Routes: `researcher -> analyst -> writer -> critic -> done`
  - LLM calls: 3, tokens in/out: 1874/794
  - Sources: 5, answer chars: 2151
  - Errors: none
- **Query:** Compare single-agent and multi-agent workflows for customer support
  - Routes: `researcher -> analyst -> writer -> critic -> done`
  - LLM calls: 3, tokens in/out: 1730/711
  - Sources: 5, answer chars: 1933
  - Errors: none
- **Query:** Summarize production guardrails for LLM agents
  - Routes: `researcher -> analyst -> writer -> critic -> done`
  - LLM calls: 3, tokens in/out: 1779/750
  - Sources: 5, answer chars: 1947
  - Errors: none

## Analysis

Multi-agent took 2.46x the baseline latency, cost +0.0013 USD more across the query set, and scored +0.94 on the heuristic quality proxy. Quality here is a heuristic (length, citation coverage, analysis pass, term coverage), not a human rubric - use peer review for the final judgement.

## Reading these numbers

The committed run used **OpenRouter** (`OPENAI_BASE_URL=https://openrouter.ai/api/v1`,
model `openai/gpt-4o-mini`) for the LLM, the **mock search corpus** for retrieval
(no `TAVILY_API_KEY`), and local JSON tracing (no LangSmith key). So:

- **Cost is real**, taken from the `usage.cost` OpenRouter reports per call - not an
  estimate from the price table.
- **Latency is real** and dominated by sequential LLM calls: the multi-agent arm makes
  three of them (researcher, analyst, writer) plus a free deterministic critic pass,
  against one for the baseline.
- **Retrieval is still mocked**, so citation coverage says "the writer cited what it was
  given", not "the sources are good". Add a Tavily key to test real retrieval.
- **Quality is a heuristic proxy** (length, citation coverage, analysis pass, query-term
  coverage), not a human judgement. Use `docs/peer_review_rubric.md` for the real score.

Measured over the 3 queries in `configs/lab_default.yaml` (the run committed in
`reports/benchmark_report.md`; latency moves roughly +/-10% between runs, the token and
cost ratios are stable):

| | single-agent | multi-agent | ratio |
|---|---:|---:|---:|
| Latency (avg, s) | 5.95 | 14.66 | 2.46x |
| Cost (total, USD) | 0.0008 | 0.0022 | 2.75x |
| Tokens (total) | 2374 | 7638 | 3.22x |
| Quality (heuristic 0-10) | 8.7 | 9.7 | +1.0 |
| Citation coverage | 100% | 100% | = |
| Failure rate | 0% | 0% | = |

## What multi-agent bought, and what it cost

| Dimension | Single-agent | Multi-agent |
|---|---|---|
| LLM calls per query | 1 | 3 (+1 free deterministic critic pass) |
| Measured price of that | 0.0008 USD / 6.0s | 0.0022 USD / 14.7s |
| Failure surface | one call fails, the run fails | any worker can fail and the run still returns a degraded answer |
| Debuggability | one opaque output | `route_history` + per-step trace attributes the failure to one agent |
| Citation discipline | prompt-only | prompt + writer post-processing + critic validation |
| Latency / cost | best | ~3x |

The honest reading: 2.5x latency and 2.6x cost bought **+1.0 point on a heuristic that
tops out at 10** - and part of that point comes from the heuristic itself rewarding the
presence of an analysis pass, which the baseline structurally cannot have. Citation
coverage was already 100% for both arms. What the split genuinely buys on this workload
is **traceability and per-step guardrails**, not answer quality. That matches the
guidance in the sources - add agents when the task genuinely decomposes, not by default.
On a task this size, a single well-prompted call is the better default; the multi-agent
version earns its keep when a step needs its own retry policy, its own model, or its own
reviewer.

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

5. **CI failed on an optional dependency.** A test asserted the OpenAI backend is
   selected when a key is present, but CI installs only the `dev` extra, so the `openai`
   SDK was missing and the test failed on the runner while passing locally.
   *Fix:* `pytest.importorskip("openai")`; the offline path stays covered.

6. **Terminal rendering hid the citations.** Rich interpreted `[1]` and `[offline-stub]`
   as markup and silently dropped them - the answer looked uncited when it was not.
   *Fix:* answers are printed via `rich.text.Text`, markup disabled.
