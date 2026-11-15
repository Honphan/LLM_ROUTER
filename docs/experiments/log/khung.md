# Nhật ký khung dùng chung

Nhật ký của phần mà mọi router dùng chung: schema và nguồn dữ liệu, tier, embedding, hiệu chỉnh ngưỡng,
đánh giá (PGR / APGR / CPT), CLI, và baseline `random`. Mỗi thay đổi một mục có ngày, mới nhất ở trên,
theo khuôn ở [`sw_ranking.md`](sw_ranking.md#khuôn-một-mục). Mạch chung của dự án: [`../LOG.md`](../LOG.md).
Quyết định có phương án bị loại ghi thành ADR ở `docs/decisions/`; ở đây chỉ tóm tắt và trỏ sang.

## Mục lục

| Ngày | Thay đổi | ADR | Ảnh hưởng tới số liệu cũ |
| --- | --- | --- | --- |
| 2026-11-14 | nguồn dữ liệu `gpt4_judge`; `check_pair` chỉ cấm hai model cùng tier | 0005 | không; thí nghiệm Arena-only giữ nguyên |
| 2026-11-11 | pipeline dữ liệu thật: tải và chuyển đổi Arena 55k, ba benchmark; embedder local | 0003 | chưa có số liệu trước đó |
| 2026-11-11 | khung cơ bản chạy end-to-end offline với router `random`; APGR hình thang | 0003, 0004 | chưa có số liệu trước đó |

## Nhật ký

### 2026-11-14 — Nguồn `gpt4_judge` và nới kiểm tra tier

#### Mục tiêu / vấn đề cần giải

Thí nghiệm Arena + judge của SW Ranking cần nạp `routellm/gpt4_judge_battles`. Khi gộp nguồn này, rating
toàn cục xếp model yếu (Mixtral, 1148, tier 0) trên model mạnh (GPT-4-1106, 922, tier 2), và
`check_pair` dừng `prepare_data` vì đòi tier model mạnh phải tốt hơn.

#### Thay đổi

- `src/llmrouter/data/sources/gpt4_judge.py` (mới): `convert_gpt4_judge`, dùng lại `convert_arena` vì cột
  thô giống Arena 55k; `source = gpt4_judge`.
- `src/llmrouter/data/sources/arena.py`: thông báo lỗi và dòng log dùng tên nguồn thay cho chữ "arena"
  cố định.
- `src/llmrouter/data/prepare.py`: thêm kind `gpt4_judge` đọc parquet, không nạp hai cột câu trả lời.
- `src/llmrouter/data/tiers.py::check_pair`: chỉ báo lỗi khi hai model cùng tier hoặc thiếu tier; khi
  model yếu ở tier tốt hơn thì `log.warning` (ADR 0005).
- `configs/data/routellm_public.yaml`: mục tải và chuyển đổi nguồn judge.
- Test: `tests/test_sources.py`, `tests/test_tiers.py`, `tests/test_sw_ranking.py`. 93 test offline xanh.
- Trạng thái git ngày 2026-11-15: các thay đổi này chưa commit.

#### Kết quả và hệ quả

- Tier và kết quả của các thí nghiệm Arena-only không đổi; không số liệu nào phải chạy lại.
- "Tier 0 = tốt nhất" từ nay chỉ đúng theo nhãn của dữ liệu fit, không còn là thứ bậc năng lực khi có
  nguồn judge.
- Kết quả thí nghiệm dùng thay đổi này: [`sw_ranking.md`](sw_ranking.md), mục 2026-11-14.

#### Hạn chế và việc tiếp theo

Phương án đối chứng "tier chỉ tính từ Arena" (bị loại trong ADR 0005 vì khác repo gốc) chưa chạy.

### 2026-11-11 — Pipeline dữ liệu thật và embedder local

Viết bù ngày 2026-11-15 từ commit `b483452`, ADR 0003, bảng dữ liệu của `../LOG.md` và
`experiments/sw_ranking_arena/`.

#### Mục tiêu / vấn đề cần giải

Khung mới chỉ chạy trên dữ liệu giả lập với embedder `hashing`. Để đo router trên MT Bench, MMLU và GSM8K
cần nạp Chatbot Arena, ba benchmark của repo RouteLLM, và một embedder có ngữ nghĩa chạy được không tốn
tiền.

#### Thay đổi

- `src/llmrouter/data/download.py`, `prepare.py`: tải dữ liệu công khai về `data/raw/` và chuyển đổi sang
  `data/processed/`.
- `src/llmrouter/data/sources/arena.py` (Arena 55k về schema chuẩn) và `routellm_bench.py` (MT Bench,
  MMLU, GSM8K về 3 cột `prompt, strong_score, weak_score`).
- `src/llmrouter/embeddings/providers.py`: provider `sentence_transformers`; sửa `cache.py`.
- `src/llmrouter/cli.py`, `pipeline.py`: lệnh `prepare-data` (tải dữ liệu công khai và chuyển đổi).
- `configs/data/routellm_public.yaml`, `configs/embeddings/bge_small.yaml`.
- Test: `tests/test_sources.py`, `tests/test_embeddings.py`, `tests/test_e2e_toy.py`.

#### Dữ liệu

- Arena 55k (`lmarena-ai/arena-human-preference-55k`): 57.477 trận thô, 55.361 sau lọc; model_a thắng
  19.335, hoà 17.120, model_b thắng 18.906. Số dòng sau lọc khớp số dòng embedding của repo gốc (55.361).
- Benchmark lấy từ repo RouteLLM, commit `0b64fdaf`. Repo gốc chưa clone; đối chiếu bằng cách đọc mã nguồn
  tại commit đó (ADR 0003, phần cập nhật).
- Chi phí: 0 USD.

#### Kết quả

Kiểm tra nhiễm chéo (cosine ≥ 0,95, embedder bge-small, dữ liệu fit là Arena) loại 8 câu MT Bench, 153 câu
MMLU, 9 câu GSM8K. Router `random` trên ba benchmark đã lọc (20 seed, CI 95%): APGR 0,485 ± 0,021 (MT Bench),
0,499 ± 0,004 (MMLU), 0,510 ± 0,009 (GSM8K), tức quanh 0,50 như kỳ vọng.

#### Hạn chế và việc tiếp theo

- Ngưỡng 0,95 là của bài báo (embedder OpenAI). Với bge-small nó loại 153 câu MMLU, trong khi repo gốc
  loại 5; cần xem lại ngưỡng theo embedder.
- Thời gian chạy và sự cố của đợt này: không có ghi chép.

### 2026-11-11 — Khung cơ bản và baseline `random`

Viết bù ngày 2026-11-15 từ commit `587e6ac`, ADR 0003 và 0004, `experiments/random_toy/`.

#### Mục tiêu / vấn đề cần giải

Repo mới chỉ có cấu hình và tài liệu. Cần một khung chạy trọn vòng `fit → calibrate → evaluate` offline,
với giao diện router đủ gọn để cắm thuật toán mới, trước khi có router thật (ADR 0003: viết khung từ đầu,
không fork repo RouteLLM).

#### Thay đổi

- `src/llmrouter/routers/base.py`: giao diện `Router` (`fit`, `strong_win_rate`, `route`, `save`, `load`);
  `registry.py`: đăng ký router theo tên.
- `src/llmrouter/routers/random/router.py`: router ngẫu nhiên làm baseline.
- `src/llmrouter/data/`: `schema.py` (schema battle và benchmark), `io.py`, `filters.py` (bỏ prompt ngắn
  hơn 16 ký tự, lấy lượt người dùng đầu tiên), `split.py` (tách tập calibrate theo prompt), `tiers.py`
  (rating toàn cục + chia 10 tier), `contamination.py` (cosine lớn nhất giữa câu benchmark và câu fit),
  `toy.py` (dữ liệu giả lập). Test của các module này nằm trong commit `587e6ac`; file nguồn thì tới
  commit `b483452` mới vào git.
- `src/llmrouter/evaluation/`: `metrics.py` (PGR, APGR, CPT), `calibrate.py` (ngưỡng α theo tỉ lệ gọi
  model mạnh), `runner.py`.
- `src/llmrouter/embeddings/`: `Embedder`, provider `hashing`, cache.
- `src/llmrouter/llm/cost.py`: chi phí token theo giá trong `configs/models/`.
- `src/llmrouter/config.py` (YAML + `include:` + `--set`), `cli.py`, `pipeline.py`.
- `configs/embeddings/hashing.yaml`, `configs/models/toy.yaml`, `configs/models/gpt4_mixtral.yaml`,
  `configs/experiments/random_toy.yaml`; `make toy`, `make test`.

#### Quyết định về APGR (ADR 0004)

Tính đúng câu chữ phương trình 8 bài báo (trung bình PGR tại 10 mức 10%…100%) thì router ngẫu nhiên ra
0,55 chứ không phải 0,50. Khung tính APGR bằng tích phân hình thang trên 11 điểm 0%…100%; đã đối chiếu
`apgr_metric` của repo gốc (dùng `np.trapz`) và giữ cách này. Khác biệt nhỏ còn lại: repo gốc lấy ngưỡng
theo phân vị điểm (`pd.qcut`), ở đây mỗi mức gọi đúng tỉ lệ và tách hoà theo thứ tự ổn định.

#### Quy trình chạy

```
make toy
```

tương đương `toy-data`, `fit`, `calibrate --pct 0.3 --pct 0.5 --pct 0.8`, `evaluate` với
`configs/experiments/random_toy.yaml`. Không cần mạng, chi phí 0.

#### Kết quả

Benchmark toy, 300 câu; dữ liệu fit: 2.000 trận giả lập (tách 10% để calibrate, 200 prompt); embedder
`hashing`.

| Thí nghiệm | APGR | CPT(50%) | CPT(80%) | Random 20 seed: APGR | Random: CPT(50%) / CPT(80%) |
| --- | --- | --- | --- | --- | --- |
| `random_toy` (seed 0) | 0,547 | 42,3% | 67,0% | 0,495 ± 0,020 | 48,8% / 79,2% |

Một lần chạy đơn của router ngẫu nhiên dao động quanh 0,50 (seed 0 cho 0,547); trung bình 20 seed là
0,495, đúng kỳ vọng. Vì vậy mọi so sánh dùng trung bình 20 seed chứ không dùng một lần chạy.

#### Hạn chế và việc tiếp theo

Khung mới được kiểm trên dữ liệu giả lập; bước tiếp theo là pipeline dữ liệu thật (mục trên) và router
tham chiếu SW Ranking ([`sw_ranking.md`](sw_ranking.md)).
