# Lab 20: Multi-Agent Research System Starter

Bài lab **Multi-Agent Systems**: hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer**, benchmark với single-agent baseline.

> Repo này là **bản đã hoàn thành** của starter skeleton: toàn bộ `TODO(student)` trong `src/` đã được implement (LLM client, search client, routing policy, 4 agent + critic, LangGraph workflow, tracing, benchmark, report).

**Chạy được không cần API key.** Khi `.env` chưa có `OPENAI_API_KEY` / `TAVILY_API_KEY`, hệ thống tự dùng backend offline deterministic (LLM stub gắn nhãn `[offline-stub]` + mock search corpus) nên test/CI và demo vẫn chạy end-to-end. Có key thì tự động chuyển sang OpenAI + Tavily, không đổi code.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Supervisor, Researcher, Analyst, Writer, Critic, Baseline
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow + fallback engine
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark + markdown report
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests for skeleton behavior
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev,llm]"
cp .env.example .env
```

### 2. Cấu hình API keys

Mở `.env` và điền key cần thiết.

```bash
OPENAI_API_KEY=...
# optional
LANGSMITH_API_KEY=...
TAVILY_API_KEY=...
```

**Dùng OpenRouter** (hoặc bất kỳ gateway OpenAI-compatible nào — Azure OpenAI, Groq,
Together, vLLM local): chỉ cần thêm `OPENAI_BASE_URL`, không sửa code.

```bash
OPENAI_API_KEY=sk-or-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini   # hoặc anthropic/claude-3.5-haiku, meta-llama/...
```

Model id giữ nguyên prefix vendor. Cost lấy trực tiếp từ `usage.cost` do OpenRouter trả
về; nếu provider không trả thì rơi về bảng giá tĩnh trong `services/llm_client.py`
(thêm model mới vào `PRICING_USD_PER_MTOK` nếu muốn số chính xác).

### 3. Chạy smoke test

```bash
make test        # 34 tests
make lint        # ruff
make typecheck   # mypy strict
python -m multi_agent_research_lab.cli --help
```

### 4. Chạy single-agent baseline

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Một agent làm tất cả trong đúng 1 LLM call — đây là arm đối chứng của benchmark.

### 5. Chạy multi-agent workflow

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary" \
  --trace-out reports/trace_multi_agent.json
```

In `final_answer`, bảng run summary (routes, tokens, cost, errors) và trace từng bước.
Cờ hữu ích: `--no-critic` (bỏ critic pass), `--no-langgraph` (chạy engine loop nội bộ),
`--json` (in raw state).

### 6. Chạy benchmark

```bash
python -m multi_agent_research_lab.cli benchmark
```

Chạy cùng query set (`configs/lab_default.yaml`) qua cả hai pipeline và ghi
`reports/benchmark_report.md` (bảng số liệu + phần phân tích tay từ `docs/benchmark_notes.md`).

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, chạy baseline skeleton | `cli.py`, `services/llm_client.py` |
| 15-45' | Build Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có benchmark report thay vì chỉ demo output đẹp.

## Đã implement

| Phần | File | Ghi chú |
|---|---|---|
| LLM client | `services/llm_client.py` | OpenAI + offline backend, retry `tenacity`, timeout, token/cost accounting |
| Search client | `services/search_client.py` | Tavily (urllib + `certifi`) + mock corpus, tự fallback khi lỗi |
| Routing policy | `agents/supervisor.py` | Deterministic `decide()`, iteration cap, per-agent retry cap |
| Worker agents | `agents/researcher.py`, `analyst.py`, `writer.py` | Prompt theo role, citation `[n]`, reference list tự append |
| Critic (bonus) | `agents/critic.py` | Kiểm citation range, độ dài, đã qua analysis chưa |
| Baseline | `agents/baseline.py` | Single-agent, 1 LLM call |
| Workflow | `graph/workflow.py` | LangGraph `StateGraph` + conditional edges; fallback loop engine; `_finalize` luôn trả câu trả lời `[degraded]` thay vì rỗng |
| Tracing | `observability/tracing.py` | Span nội bộ + LangSmith/Langfuse khi có key + export JSON |
| Benchmark | `evaluation/benchmark.py`, `report.py` | latency, cost, quality heuristic, citation coverage, failure rate |

Guardrails: `max_iterations`, `timeout_seconds`, retry có backoff, skip agent hỏng sau 2 lần
fail, LangGraph `recursion_limit`, Pydantic validation, critic pass.

Thiết kế chi tiết: `docs/design_template.md`.

## Deliverables

| Yêu cầu | Trong repo |
|---|---|
| Code hoàn chỉnh, lint/test pass | `src/`, `tests/` (34 tests) |
| Trace evidence | `reports/trace_multi_agent.json` (dùng LangSmith/Langfuse nếu có key) |
| Benchmark report | `reports/benchmark_report.md` |
| Failure mode + cách fix | `docs/benchmark_notes.md` |
| Design doc | `docs/design_template.md` |
| Exit ticket | `docs/exit_ticket.md` |

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
