# Exit Ticket

## 1. Case nào nên dùng multi-agent? Vì sao?

Khi task **tách được thành các bước có tiêu chí chất lượng khác nhau** và ta cần biết bước
nào hỏng. Ví dụ của lab: tìm nguồn (đo bằng độ phủ và độ tin cậy nguồn) — đánh giá bằng
chứng (đo bằng việc phát hiện mâu thuẫn) — viết cho một đối tượng cụ thể (đo bằng độ rõ và
citation). Ba tiêu chí đó xung đột nhau trong một prompt duy nhất.

Cụ thể, nên tách khi có ít nhất một trong các dấu hiệu:

- Cần **quy trách nhiệm lỗi**: trace theo agent cho biết sai ở search hay ở khâu viết.
- Cần **guardrail khác nhau cho từng bước**: researcher được retry, writer thì degrade.
- Các bước **chạy song song được** (nhiều chủ đề con) hoặc dùng **model/công cụ khác nhau**
  (model rẻ cho phân loại, model mạnh cho tổng hợp).
- Output cần **kiểm tra độc lập** — critic không nên là chính agent đã viết ra câu trả lời.

## 2. Case nào không nên dùng multi-agent? Vì sao?

Khi task **một prompt làm được**: hỏi đáp ngắn, trích xuất field, phân loại, viết lại văn
bản, hoặc luồng có latency SLA chặt (chat realtime). Ở đó multi-agent trả giá bằng:

- **~3x LLM call** → 3x cost và latency cộng dồn tuần tự (xem `docs/benchmark_notes.md`).
- **Mất context qua handoff**: mỗi lần chuyển bàn là một lần tóm tắt lossy.
- **Nhiều điểm hỏng hơn**: router chọn sai, worker trả rỗng, vòng lặp supervisor.
- **Khó chấm**: chất lượng cuối là tổng hợp của nhiều bước, khó gán nguyên nhân nếu
  không có trace tử tế.

Nguyên tắc rút ra: bắt đầu bằng single-agent baseline, **đo**, và chỉ tách agent khi số
liệu cho thấy một bước cụ thể đang là nút thắt chất lượng — không tách vì kiến trúc trông
"xịn" hơn.
