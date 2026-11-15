# Lộ trình

Cập nhật khi một router đổi trạng thái hoặc khi có kết quả thí nghiệm mới (mạch dự án và bảng kết quả ở
`docs/experiments/LOG.md`, chi tiết từng thuật toán ở `docs/experiments/log/`). Giữ file ngắn: nó được nạp vào mọi phiên qua CLAUDE.md.

## Giai đoạn

RouteLLM (bài báo + repo gốc ở `third_party/RouteLLM`) là nền tham chiếu; đóng góp chính là thuật toán
tự phát triển trên khung này (ADR 0003).

0. **Cấu trúc cơ bản** — khung code chạy end-to-end offline với router `random`. Xong.
1. **SW Ranking** — router tham chiếu, đối chiếu repo gốc; tái hiện Bảng 1–3 trên MT Bench, MMLU, GSM8K. ← đang làm
2. **Thuật toán tự phát triển** — công cụ so sánh, phân tích lỗi của baseline, router mới.
3. **Dữ liệu tiếng Việt trong miền** — gán nhãn, augmentation, đo lại các router.
4. **Matrix Factorization / BERT / Causal LLM** — baseline bổ sung; BERT và Causal LLM cần GPU thuê.

## Trạng thái router

| Router | Trạng thái | Tài liệu | Ghi chú |
| --- | --- | --- | --- |
| `random` | xong, có test | — | baseline, APGR ≈ 0.50; chạy trọn vòng trên toy |
| `sw_ranking` | xong, có test | `docs/experiments/sw_ranking.md`, log: `docs/experiments/log/sw_ranking.md` | bge-small, APGR MT Bench / MMLU / GSM8K — Arena: 0.645 / 0.549 / 0.509; Arena + judge: 0.792 / 0.510 / 0.574 |
| `matrix_factorization` | chưa bắt đầu | — | GPU 8GB hoặc CPU |
| `bert` | chưa bắt đầu | — | GPU 24GB |
| `causal_llm` | chưa bắt đầu | — | GPU lớn |

## Việc còn dở của giai đoạn 1

- Chạy lại với `text-embedding-3-small` (cần khóa OpenAI, ~$0.05) để so trực tiếp với Bảng 1–3.
- Quét `gamma` cho embedder local: ngưỡng α ở 20%/50%/80% chỉ cách nhau ~0.002, tức trọng số gần phẳng.
- Arena + judge đã chạy (2026-11-14, ghi ở `docs/experiments/LOG.md`): MMLU tụt xuống gần
  random, cần tìm nguyên nhân; chạy đối chứng "tier chỉ tính từ Arena" (ADR 0005).
- Ngưỡng nhiễm chéo 0.95 với bge-small loại 153 câu MMLU (repo gốc loại 5): cần xem lại theo embedder.

## Câu hỏi mở

- Embedder nào cho câu hỏi tiếng Việt: `text-embedding-3-small` hay `bge-m3`?
- Cặp model đích cho sản phẩm (ví dụ qwen3.5-27b / qwen3.5-9b) và giá token thực tế.
