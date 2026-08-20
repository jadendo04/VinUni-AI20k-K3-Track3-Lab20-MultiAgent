# Design: Multi-Agent Research System

## Problem

Nhận một câu hỏi nghiên cứu mở (ví dụ "Research GraphRAG state-of-the-art and write a
500-word summary"), tìm nguồn, đánh giá bằng chứng, và viết câu trả lời có trích dẫn cho
đối tượng `technical learners`. Đầu ra phải kèm trace để debug và số liệu để benchmark.

## Why multi-agent?

Single-agent vẫn làm được task này (và là baseline trong repo). Tách agent chỉ đáng khi:

- Ba việc *search / đánh giá / viết* có tiêu chí chất lượng khác nhau; nhồi cả ba vào một
  prompt làm loãng context và khó biết bước nào hỏng.
- Cần quy trách nhiệm khi sai: `route_history` + trace chỉ đúng agent gây lỗi.
- Cần guardrail riêng cho từng bước (researcher fail thì skip, writer fail thì degrade).

Đổi lại: gấp ~3 lần số LLM call, latency và cost tăng tương ứng. Xem
`docs/benchmark_notes.md`.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Đọc state, chọn route kế tiếp hoặc dừng | toàn bộ `ResearchState` | `next_route`, `route_history` | route vòng lặp vô hạn → chặn bằng `max_iterations` + failure counter |
| Researcher | Gọi search, viết `research_notes` có `[n]` | `request` | `sources`, `research_notes` | search rỗng/timeout → `AgentExecutionError`, retry tối đa 2 lần rồi skip |
| Analyst | So sánh claim, đánh dấu bằng chứng yếu | `research_notes` | `analysis_notes` | thiếu `research_notes` → fail sớm, không bịa |
| Writer | Tổng hợp thành câu trả lời cuối có citation | `research_notes`, `analysis_notes`, `sources` | `final_answer` | citation sai số hiệu / trống → critic bắt, reference list được append tự động |
| Critic (optional) | Validate citation range, độ dài, đã qua analysis chưa | `final_answer`, `sources` | verdict + `citation_coverage` | không có `final_answer` → fail, workflow trả bản `[degraded]` |

## Shared state

`src/multi_agent_research_lab/core/state.py`:

| Field | Vì sao cần |
|---|---|
| `request` | input gốc; `max_sources`, `audience` chi phối prompt |
| `iteration`, `route_history`, `next_route` | guardrail + bằng chứng thứ tự handoff khi debug |
| `sources` | citation phải trỏ về đúng danh sách này |
| `research_notes`, `analysis_notes` | điều kiện routing của supervisor và context của writer |
| `final_answer` | deliverable |
| `agent_results` | token/cost/metadata từng bước → benchmark đọc trực tiếp |
| `trace` | timeline từng span, export JSON làm evidence |
| `errors`, `failed_agents` | fallback policy: skip agent hỏng thay vì retry vô hạn |

## Routing policy

```text
supervisor -> researcher -> supervisor -> analyst -> supervisor -> writer
           -> supervisor -> critic -> supervisor -> done
```

Deterministic, suy ra hoàn toàn từ state (`SupervisorAgent.decide`):

1. `iteration >= max_iterations` → `done`
2. có `final_answer` → `critic` (nếu bật, chạy đúng một lần) → `done`
3. chưa có `research_notes` → `researcher`
4. có `research_notes`, chưa có `analysis_notes` → `analyst`
5. có notes bất kỳ → `writer`
6. mọi agent còn lại đã vượt retry → `done`

Không dùng LLM router: quyết định biểu diễn được bằng rule nên router LLM chỉ thêm
latency, cost và nondeterminism.

## Guardrails

- **Max iterations:** `Settings.max_iterations` (mặc định 6) trong `decide()`, cộng
  `recursion_limit = max_iterations * 2 + 4` phía LangGraph.
- **Timeout:** `Settings.timeout_seconds` truyền vào OpenAI client và `urlopen` của search.
- **Retry:** `tenacity` 3 lần, exponential backoff, trong `LLMClient._complete_openai`;
  search fail → tự fallback sang mock corpus.
- **Fallback:** `BaseAgent.execute` nuốt exception thành `state.record_failure`;
  `MultiAgentWorkflow._finalize` luôn trả câu trả lời `[degraded]` thay vì rỗng.
- **Validation:** Pydantic schema cho mọi input/output; `CriticAgent` kiểm citation range,
  độ dài, và việc đã qua analysis.

## Benchmark plan

- **Query set:** 3 query trong `configs/lab_default.yaml`.
- **Arms:** `single-agent` (`SingleAgentBaseline`, 1 LLM call) vs `multi-agent`.
- **Metrics:** latency (wall-clock), cost (token × bảng giá), quality (heuristic 0-10),
  citation coverage, failure rate.
- **Expected outcome:** multi-agent thắng về citation coverage + khả năng debug, thua về
  latency/cost (~3x). Kết quả thực tế và failure mode: `reports/benchmark_report.md`.
