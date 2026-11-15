# Nhật ký dự án

Log chung của llm-router: mạch phát triển (bước trước vướng gì → giải thế nào → kết quả ra sao), bảng kết
quả tổng hợp và bảng dữ liệu. Dùng làm xương sống khi viết báo cáo; chi tiết từng bước nằm ở log riêng.

## Cách ghi

- **File này**: mỗi mốc đáng kể (router mới, nguồn dữ liệu mới, biến thể làm đổi kết luận) thêm một mục vào
  "Dòng thời gian"; mỗi lần `evaluate` có ý nghĩa thêm một dòng vào "Bảng kết quả tổng hợp"; mỗi nguồn dữ
  liệu mới thêm một dòng vào "Dữ liệu". Mới nhất ở trên.
- **Log riêng** ở `log/`: một file cho mỗi thuật toán ([`log/sw_ranking.md`](log/sw_ranking.md)) và một file
  cho phần dùng chung ([`log/khung.md`](log/khung.md)). Mọi thay đổi code, cấu hình, dữ liệu hay lần chạy
  thí nghiệm đều có một mục ở đó: mục tiêu, thay đổi, lệnh chạy, sự cố, kết quả, nhận xét, hạn chế.
- **Quyết định** có phương án bị loại ghi thành ADR ở [`../decisions/`](../decisions/README.md); log chỉ
  tóm tắt và trỏ sang.
- Số liệu chép từ `experiments/<tên>/eval/<benchmark>/summary.json` (do CLI ghi), luôn kèm router `random`
  trên cùng benchmark và seed. Mốc bài báo lấy từ `.claude/skills/compare-experiments/paper_baselines.md`.
  Ghi lệnh `--set` đầy đủ nếu có.

## Dòng thời gian

Cặp model ở mọi thí nghiệm thật: gpt-4-1106-preview (mạnh) / mixtral-8x7b-instruct-v0.1 (yếu). Random là
trung bình ± CI 95% trên 20 seed, cùng benchmark đã lọc nhiễm chéo.

### 2026-11-14 — SW Ranking với dữ liệu Arena + judge

- **Vấn đề của bước trước.** SW Ranking chỉ fit trên Arena cho `P(strong)` gần như hằng số (0,695–0,701)
  và ngang random trên GSM8K: Arena chỉ có 3.757 trận ở cặp tier của hai model đích và ít câu kiểu GSM8K.
- **Cách giải quyết.** Làm theo mục 4.1.2 bài báo: gộp thêm 109.101 trận gán nhãn bằng GPT-4 judge
  (`routellm/gpt4_judge_battles`), nâng số trận ở cặp tier đích lên khoảng 110 nghìn. Siêu tham số giữ
  nguyên, không tham số nào chọn theo benchmark.
- **Vướng mắc khi làm.** 68,5% nhãn judge nghiêng về model yếu nên rating toàn cục xếp Mixtral trên GPT-4,
  và kiểm tra tier của khung từ chối dữ liệu. Giải bằng ADR 0005: tier tính trên dữ liệu gộp như repo gốc,
  chỉ cấm hai model cùng tier. Hai lần chạy hỏng vì bộ nhớ (GPU khi embed, RAM khi suy luận), sửa bằng
  kích thước lô trong cấu hình.
- **Kết quả.**

  | Benchmark | Dữ liệu fit | APGR | CPT(50%) | CPT(80%) | Random APGR | Random CPT(50%) / CPT(80%) | So với Arena-only trên tập câu chung (CI 95%) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | MT Bench, 73 câu | Arena + judge | 0,792 | 13,7% | 38,4% | 0,506 ± 0,022 | 49,3% / 78,4% | +0,144 [−0,011, +0,278] |
  | MMLU, 13.766 câu | Arena + judge | 0,510 | 46,2% | 76,0% | 0,501 ± 0,004 | 49,8% / 80,2% | −0,038 [−0,061, −0,014] |
  | GSM8K, 1.301 câu | Arena + judge | 0,574 | 41,1% | 68,1% | 0,510 ± 0,005 | 48,0% / 78,7% | +0,067 [+0,014, +0,121] |

  MT Bench và GSM8K tăng và sát dòng "Arena + judge" của bài báo (0,759 và 0,568) dù embedder khác.
- **Vấn đề còn lại.** MMLU tụt về gần random và kém bản Arena-only, chưa rõ nguyên nhân. `P(strong)` vẫn
  rất hẹp (ba ngưỡng α cách nhau cỡ 0,003). Một điểm số chỉ dùng log độ dài prompt còn hơn SW Ranking trên
  MMLU và GSM8K (kiểm tra nhanh ngoài CLI).
- **Chi tiết.** [`log/sw_ranking.md`](log/sw_ranking.md) mục 2026-11-14; [`log/khung.md`](log/khung.md) mục
  2026-11-14; [ADR 0005](../decisions/0005-tier-order-with-judge-data.md).

### 2026-11-11 — SW Ranking, router tham chiếu đầu tiên, fit trên Arena

- **Vấn đề của bước trước.** Khung mới chỉ có router `random` và dữ liệu giả lập; chưa có router thật nào
  để kiểm chứng pipeline và làm mốc so sánh.
- **Cách giải quyết.** Cài SW Ranking theo mục 4.2 bài báo (phương trình 9–10): không huấn luyện, giải
  Bradley-Terry có trọng số theo độ giống lúc suy luận. Nạp Chatbot Arena 55k và ba benchmark của repo
  RouteLLM. Dùng embedder local `bge-small-en-v1.5` để không tốn tiền (bài báo dùng `text-embedding-3-small`).
- **Kết quả.**

  | Benchmark | Dữ liệu fit | APGR | CPT(50%) | CPT(80%) | Random APGR | Random CPT(50%) / CPT(80%) | Bài báo (APGR) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | toy, 300 câu | toy 1.800 trận | 0,833 | 19,0% | 33,3% | 0,495 ± 0,020 | 48,8% / 79,2% | — |
  | MT Bench, 72 câu | Arena 55k | 0,645 | 41,7% | 54,2% | 0,485 ± 0,021 | 50,5% / 80,6% | 0,610 |
  | MMLU, 13.889 câu | Arena 55k | 0,549 | 42,3% | 75,9% | 0,499 ± 0,004 | 49,8% / 80,1% | 0,473 |
  | GSM8K, 1.310 câu | Arena 55k | 0,509 | 49,9% | 79,9% | 0,510 ± 0,009 | 48,6% / 79,0% | 0,475 |

  Hơn random rõ trên MT Bench và MMLU; ngang random trên GSM8K.
- **Vấn đề còn lại.** `P(strong)` gần hằng số: ngưỡng α ở 20% / 50% / 80% chỉ cách nhau khoảng 0,002, tức
  trọng số các trận gần phẳng với `gamma = 10`. GSM8K không có tín hiệu. Embedder khác bài báo nên chưa so
  trực tiếp được. Đây là lý do của bước 2026-11-14.
- **Chi tiết.** [`log/sw_ranking.md`](log/sw_ranking.md) mục 2026-11-11; [`log/khung.md`](log/khung.md) mục
  "Pipeline dữ liệu thật"; giải thích thuật toán ở [`sw_ranking.md`](sw_ranking.md).

### 2026-11-11 — Khung cơ bản chạy end-to-end với router `random`

- **Vấn đề của bước trước.** Repo chỉ có cấu hình và tài liệu, chưa có code. Fork repo RouteLLM thì nhanh
  có số liệu nhưng giao diện router, cấu hình và đánh giá của nó khó cắm thuật toán mới và khó kiểm thử
  offline.
- **Cách giải quyết.** Viết khung `llmrouter` từ đầu (ADR 0003): một giao diện `Router`, một schema dữ
  liệu, cấu hình YAML, vòng đời `fit → calibrate → evaluate` qua CLI, dữ liệu giả lập và embedder `hashing`
  để chạy không cần mạng. APGR tính bằng tích phân hình thang, khớp repo gốc (ADR 0004: tính theo câu chữ
  phương trình 8 thì random ra 0,55 thay vì 0,50).
- **Kết quả.** `random_toy` trên benchmark toy 300 câu (fit trên 2.000 trận giả lập): APGR 0,547,
  CPT(50%) 42,3%, CPT(80%) 67,0% với seed 0; trung bình 20 seed là APGR 0,495 ± 0,020, CPT(50%) 48,8%,
  CPT(80%) 79,2%, đúng kỳ vọng ≈ 0,50 của router ngẫu nhiên.
- **Vấn đề còn lại.** Khung mới được kiểm trên dữ liệu giả lập, chưa có router thật.
- **Chi tiết.** [`log/khung.md`](log/khung.md) mục "Khung cơ bản";
  [ADR 0003](../decisions/0003-routellm-as-baseline.md), [ADR 0004](../decisions/0004-apgr-trapezoid.md).

### 2026-11-08 — Chọn điểm xuất phát: SW Ranking trước các router có tham số

- **Bối cảnh.** Cần một router chạy được trên máy 8GB RAM không GPU để dựng và kiểm chứng toàn bộ pipeline.
- **Quyết định.** Làm SW Ranking trước Matrix Factorization: MF có kết quả tốt hơn trong bài báo nhưng cần
  vòng huấn luyện và GPU để lặp nhanh. Phần dùng chung dựng cho SW Ranking được các router sau dùng lại.
- **Chi tiết.** [ADR 0001](../decisions/0001-sw-ranking-first.md).

## Bảng kết quả tổng hợp

Mỗi lần `evaluate` có ý nghĩa một dòng, mới nhất ở trên.

| Ngày | Thí nghiệm | Router | Dữ liệu fit | Benchmark | APGR | CPT(50%) | CPT(80%) | Bài báo (APGR) | Nhận xét | Log riêng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-14 | sw_ranking_arena_judge (`--set eval.benchmark_name=gsm8k --set eval.benchmark_path=data/processed/bench_gsm8k.parquet`) | sw_ranking, bge-small-en-v1.5 | Arena 55k + judge 109k (fit 156.258 trận) | GSM8K, 1.301 câu (loại 18) | 0.574 (random 0.510 ± 0.005) | 41.1% (random 48.0%) | 68.1% (random 78.7%) | 0.568 | Hơn random và hơn Arena-only (0.509) rõ; sát bài báo. Trên 1.300 câu chung: 0.574 so với 0.507, CI 95% của hiệu [+0.014, +0.121] | [sw_ranking, 2026-11-14](log/sw_ranking.md) |
| 2026-11-14 | sw_ranking_arena_judge (`--set eval.benchmark_name=mmlu --set eval.benchmark_path=data/processed/bench_mmlu.parquet`) | sw_ranking, bge-small-en-v1.5 | Arena 55k + judge 109k | MMLU, 13.766 câu (loại 276) | 0.510 (random 0.501 ± 0.004) | 46.2% (random 49.8%) | 76.0% (random 80.2%) | — (bài báo chỉ có Arena + gold: 0.603) | Gần random và **kém Arena-only** (0.549). Trên 13.765 câu chung: 0.510 so với 0.548, CI 95% của hiệu [−0.061, −0.014] | [sw_ranking, 2026-11-14](log/sw_ranking.md) |
| 2026-11-14 | sw_ranking_arena_judge | sw_ranking, bge-small-en-v1.5 | Arena 55k + judge 109k | MT Bench, 73 câu (loại 7) | 0.792 (random 0.506 ± 0.022) | 13.7% (random 49.3%) | 38.4% (random 78.4%) | 0.759 | Hơn random rất rõ, sát bài báo. So với Arena-only (0.645) trên 72 câu chung: 0.789, CI 95% của hiệu [−0.011, +0.278], chỉ 72 câu nên chưa chắc chắn. Chi tiết ở log riêng | [sw_ranking, 2026-11-14](log/sw_ranking.md) |
| 2026-11-11 | sw_ranking_arena (`--set eval.benchmark_name=mmlu --set eval.benchmark_path=data/processed/bench_mmlu.parquet`) | sw_ranking, bge-small-en-v1.5 | Arena 55k (fit 52.571 trận) | MMLU, 13.889 câu (loại 153) | 0.549 (random 0.499 ± 0.004) | 42.3% (random 49.8%) | 75.9% (random 80.1%) | 0.473 | Hơn random rõ; bài báo dùng embedder OpenAI nên không so trực tiếp | [sw_ranking, 2026-11-11](log/sw_ranking.md) |
| 2026-11-11 | sw_ranking_arena (`--set eval.benchmark_name=gsm8k --set eval.benchmark_path=data/processed/bench_gsm8k.parquet`) | sw_ranking, bge-small-en-v1.5 | Arena 55k | GSM8K, 1.310 câu (loại 9) | 0.509 (random 0.510 ± 0.009) | 49.9% (random 48.6%) | 79.9% (random 79.0%) | 0.475 | Ngang random, giống nhận định của bài báo (Arena ít câu kiểu GSM8K) | [sw_ranking, 2026-11-11](log/sw_ranking.md) |
| 2026-11-11 | sw_ranking_arena | sw_ranking, bge-small-en-v1.5 | Arena 55k | MT Bench, 72 câu (loại 8) | 0.645 (random 0.485 ± 0.021) | 41.7% (random 50.5%) | 54.2% (random 80.6%) | 0.610 | Hơn random rõ; chỉ 72 câu nên nhiễu lớn | [sw_ranking, 2026-11-11](log/sw_ranking.md) |
| 2026-11-11 | sw_ranking_toy | sw_ranking, hashing | toy 1.800 trận | toy, 300 câu | 0.833 (random 0.495 ± 0.020) | 19.0% | 33.3% | — | Kiểm tra pipeline offline | [sw_ranking, 2026-11-11](log/sw_ranking.md) |

## Dữ liệu

| Ngày | Nguồn | Số trận | Mạnh thắng / hoà / yếu thắng | Chi phí | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| 2026-11-14 | `routellm/gpt4_judge_battles` (revision `2a1afe8d`) | 109.101 thô, 109.101 sau lọc ≥16 ký tự | GPT-4 thắng 10.136 / hoà 24.237 / Mixtral thắng 74.728 | 0 | Chỉ một cặp model (gpt-4-1106 đấu mixtral-8x7b), nhãn do GPT-4 chấm. 2.017 prompt trùng nguyên văn với Arena. Gộp với Arena thì tier đảo chiều (ADR 0005) |
| 2026-11-11 | `lmarena-ai/arena-human-preference-55k` | 57.477 thô, 55.361 sau lọc ≥16 ký tự | model_a thắng 19.335 / hoà 17.120 / model_b thắng 18.906 | 0 | Khớp số dòng embedding của repo gốc (55.361). Benchmark MT Bench / MMLU / GSM8K lấy từ repo RouteLLM commit `0b64fdaf` |

## Vấn đề mở

Những điểm chưa giải, là đầu vào cho bước tiếp theo (việc cụ thể theo dõi ở [`../ROADMAP.md`](../ROADMAP.md)).

1. **`P(strong)` gần phẳng.** Với `gamma = 10` và embedder local, các ngưỡng α chỉ cách nhau 0,002–0,003.
   Cần quét `gamma` và thử chỉ dùng top-k trận giống nhất.
2. **MMLU tụt khi thêm dữ liệu judge** (0,549 → 0,510). Giả thuyết: tín hiệu của bản Arena-only một phần
   đến từ độ dài prompt và mất đi khi dữ liệu judge chi phối. Chưa kiểm tra.
3. **Đối chứng tier.** Chưa chạy phương án "tier chỉ tính từ Arena" (bị loại trong ADR 0005).
4. **Embedder.** Chưa chạy `text-embedding-3-small` (cần khóa OpenAI, khoảng 0,05 USD) nên chưa tái hiện
   trực tiếp Bảng 1–3 bài báo.
5. **Ngưỡng nhiễm chéo.** 0,95 với bge-small loại 153–276 câu MMLU, repo gốc loại 5; cần hiệu chỉnh theo
   embedder.
6. **Baseline độ dài prompt.** Điểm số `log(độ dài)` cho APGR 0,552 / 0,581 / 0,598 trên MT Bench / MMLU /
   GSM8K (kiểm tra nhanh ngoài CLI), hơn SW Ranking Arena + judge ở MMLU và GSM8K. Cần làm thành baseline
   chính thức trước khi phát triển thuật toán mới.
7. **Tập calibrate.** Với dữ liệu gộp, khoảng hai phần ba tập calibrate là prompt judge; repo gốc calibrate
   trên Arena.
