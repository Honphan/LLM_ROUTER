# Nhật ký SW Ranking

Nhật ký riêng của router `sw_ranking`: mỗi biến thể hoặc thay đổi (dữ liệu fit, embedder, siêu tham số,
công thức) là một mục có ngày, mới nhất ở trên. Giải thích thuật toán: [`../sw_ranking.md`](../sw_ranking.md).
Mạch chung của dự án và bảng kết quả tổng hợp: [`../LOG.md`](../LOG.md). Thay đổi phần dùng chung (dữ
liệu, tier, đánh giá): [`khung.md`](khung.md).

Số liệu chép từ `experiments/<tên>/` (do CLI ghi); phần tính thêm ngoài CLI được nói rõ trong từng mục.

## Mục lục

| Ngày | Biến thể | Thí nghiệm | APGR MT Bench / MMLU / GSM8K | Kết luận ngắn |
| --- | --- | --- | --- | --- |
| 2026-11-14 | thêm dữ liệu judge vào dữ liệu fit | `sw_ranking_arena_judge` | 0,792 / 0,510 / 0,574 | MT Bench và GSM8K tăng, sát bài báo; MMLU tụt về gần random |
| 2026-11-11 | cài đặt gốc, fit trên Arena | `sw_ranking_arena`, `sw_ranking_toy` | 0,645 / 0,549 / 0,509 | hơn random trên MT Bench và MMLU, ngang random trên GSM8K; `P(strong)` gần hằng số |

## Khuôn một mục

Mỗi mục mới dùng đủ các phần sau, phần nào không có thì ghi "không có":

- **Mục tiêu / vấn đề cần giải**: bước trước vướng gì, mục này thử điều gì.
- **Thay đổi**: code, cấu hình, dữ liệu (kèm đường dẫn file).
- **Quy trình chạy**: lệnh đầy đủ, kể cả `--set`; chi phí, thời gian, máy.
- **Sự cố**: lần chạy hỏng, nguyên nhân, cách sửa.
- **Kết quả**: benchmark, dữ liệu fit, APGR, CPT(50%), CPT(80%), kèm random cùng benchmark và seed.
- **Nhận xét**: điều rút ra, so với mục trước và với bài báo.
- **Hạn chế và việc tiếp theo**.

## Nhật ký

### 2026-11-14 — SW Ranking với dữ liệu Arena + judge

Thí nghiệm `sw_ranking_arena_judge`, so với `sw_ranking_arena`. Phần "tập câu chung" tính lại từ hai file
`per_question.parquet` bằng `llmrouter.evaluation.metrics`, không ghi vào `experiments/`.

#### Mục tiêu

Router `sw_ranking` chỉ fit trên Chatbot Arena cho `P(strong)` gần như hằng số (0,695–0,701) và ngang
random trên GSM8K. Bài báo RouteLLM (mục 4.1.2, Bảng 1 và 3) bổ sung dữ liệu gán nhãn bằng LLM judge và
báo cáo cải thiện lớn cho SW Ranking. Thí nghiệm này tái hiện bước đó với embedder local.

#### Dữ liệu

| Nguồn | Số trận | Cặp model | Nhãn |
| --- | --- | --- | --- |
| Arena 55k (`lmarena-ai/arena-human-preference-55k`) | 55.361 sau lọc | 64 model, mọi cặp | người dùng bình chọn |
| Judge (`routellm/gpt4_judge_battles`, revision `2a1afe8d`, Apache-2.0) | 109.101 (không dòng nào bị lọc) | chỉ gpt-4-1106-preview đấu mixtral-8x7b-instruct-v0.1 | GPT-4 chấm |

- Nhãn judge: GPT-4 thắng 10.136 (9,3%), hoà 24.237 (22,2%), Mixtral thắng 74.728 (68,5%).
  "Mixtral thắng" nghĩa là câu trả lời của model yếu được chấm đủ tốt, không phải thắng trong so kè
  trực tiếp như phiếu bầu Arena.
- Prompt judge dài hơn Arena: trung vị 139 ký tự (Arena 79), phân vị 90 là 1.132 ký tự (Arena 662).
- 2.017 prompt judge trùng nguyên văn với prompt Arena.
- Sau khi gộp và lọc ≥ 16 ký tự: 164.462 trận. Tách 5% prompt để calibrate: 156.258 trận fit, 8.204 trận
  (7.828 prompt) calibrate. Router dùng 144.232 trận khác tier; 12.026 trận cùng tier bị bỏ.

#### Vấn đề gặp phải: tier đảo chiều

Rating toàn cục (Bradley-Terry) tính trên dữ liệu gộp, đúng như repo gốc RouteLLM:

| | Chỉ Arena | Arena + judge |
| --- | --- | --- |
| gpt-4-1106-preview | 1165, tier 0 | 922, tier 2 |
| mixtral-8x7b-instruct-v0.1 | 1026, tier 2 | 1148, tier 0 (một mình) |
| Số trận ở cặp tier của hai model | 3.757 | 110.169 |

Vì 68,5% nhãn judge nghiêng về Mixtral, model yếu được xếp trên model mạnh. Pipeline cũ từ chối trường
hợp này (`check_pair` đòi tier model mạnh tốt hơn). Quyết định (ADR 0005): làm theo repo gốc, tier tính
trên dữ liệu gộp, chỉ cấm hai model cùng tier. Công thức `P(strong) = p_s / (p_s + p_w)` không phụ
thuộc thứ tự tier; hệ quả là `P(strong)` nằm quanh 0,22 thay vì 0,70, còn thứ hạng câu hỏi và ngưỡng α
(từ `calibrate`) vẫn dùng như cũ.

#### Thay đổi trong code

- `src/llmrouter/data/sources/gpt4_judge.py`: chuyển đổi nguồn judge về schema chuẩn, dùng lại
  `convert_arena` vì cột thô giống hệt Arena 55k; `source = gpt4_judge`.
- `src/llmrouter/data/prepare.py`: kind `gpt4_judge` đọc parquet, không nạp hai cột câu trả lời.
- `src/llmrouter/data/tiers.py::check_pair`: chỉ báo lỗi khi cùng tier; tier đảo thì cảnh báo trong log.
- `configs/data/routellm_public.yaml`: thêm mục tải và chuyển đổi.
- `configs/experiments/sw_ranking_arena_judge.yaml`: hai nguồn, trọng số 1.0 mỗi nguồn (như repo gốc).
- Test: chuyển đổi judge (5 dòng tự viết), kind parquet, `check_pair` với tier đảo, và router cho cùng
  xác suất khi đổi nhãn tier. 93 test offline xanh.

#### Quy trình chạy

```
python -m llmrouter prepare-data --config configs/experiments/sw_ranking_arena_judge.yaml
python -m llmrouter info        --config configs/experiments/sw_ranking_arena_judge.yaml
python -m llmrouter fit         --config configs/experiments/sw_ranking_arena_judge.yaml
python -m llmrouter calibrate   --config configs/experiments/sw_ranking_arena_judge.yaml --pct 0.2 --pct 0.5 --pct 0.8
python -m llmrouter evaluate    --config configs/experiments/sw_ranking_arena_judge.yaml
python -m llmrouter evaluate    --config ... --set eval.benchmark_name=mmlu  --set eval.benchmark_path=data/processed/bench_mmlu.parquet
python -m llmrouter evaluate    --config ... --set eval.benchmark_name=gsm8k --set eval.benchmark_path=data/processed/bench_gsm8k.parquet
```

- Chi phí: 0 USD, không gọi API. Tải 167 MB. Máy: RTX 3050 Laptop 4 GB.
- Thời gian: embed 148.728 prompt khoảng 7 phút; calibrate 1 phút; evaluate 0,5 phút (MT Bench),
  2,5 phút (MMLU), 1 phút (GSM8K).
- Hai lần chạy hỏng trước khi thành công, đều do bộ nhớ, không ghi ra kết quả nào:
  1. GPU hết bộ nhớ khi embed với lô 64 (prompt judge dài) → `embedder.params.batch_size: 16`.
  2. RAM không đủ khi suy luận với lô 256 câu × 144 nghìn trận → `router.params.batch_size: 64`, fit lại.
  Cả hai giá trị nằm trong file cấu hình thí nghiệm. Kích thước lô không đổi tên embedder hay khoá cache.
- Siêu tham số giữ nguyên như thí nghiệm Arena-only (`gamma = 10`, `normalize_by_max = true`,
  `prior = 0`); không tham số nào được chọn theo benchmark.

#### Kết quả

Cặp model: gpt-4-1106-preview / mixtral-8x7b-instruct-v0.1. Embedder: `BAAI/bge-small-en-v1.5`.
Random: trung bình ± CI 95% trên 20 seed, cùng benchmark đã lọc. Ngưỡng nhiễm chéo cosine ≥ 0,95.

Số liệu chính (từ CLI):

| Benchmark | Dữ liệu fit | Số câu (loại) | APGR | CPT(50%) | CPT(80%) | Random APGR | Random CPT(50%) / CPT(80%) | Bài báo: APGR, CPT(50%), CPT(80%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MT Bench | Arena | 72 (8) | 0,645 | 41,7% | 54,2% | 0,485 ± 0,021 | 50,5% / 80,6% | 0,610, 37,85%, 58,99% |
| MT Bench | Arena + judge | 73 (7) | **0,792** | 13,7% | 38,4% | 0,506 ± 0,022 | 49,3% / 78,4% | 0,759, 23,21%, 36,04% |
| MMLU | Arena | 13.889 (153) | 0,549 | 42,3% | 75,9% | 0,499 ± 0,004 | 49,8% / 80,1% | 0,473, 55,82%, 80,25% |
| MMLU | Arena + judge | 13.766 (276) | 0,510 | 46,2% | 76,0% | 0,501 ± 0,004 | 49,8% / 80,2% | không có dòng tương ứng |
| GSM8K | Arena | 1.310 (9) | 0,509 | 49,9% | 79,9% | 0,510 ± 0,009 | 48,6% / 79,0% | 0,475, 54,43%, 82,11% |
| GSM8K | Arena + judge | 1.301 (18) | **0,574** | 41,1% | 68,1% | 0,510 ± 0,005 | 48,0% / 78,7% | 0,568, 41,21%, 72,20% |

Bảng 2 bài báo dùng "Arena + gold" cho MMLU (APGR 0,603), không phải Arena + judge, nên không so được.

Quy đổi chi phí theo giá trong `configs/models/gpt4_mixtral.yaml`, so với gửi hết cho model mạnh:
MT Bench tiết kiệm 6,9 lần tại CPT(50%) và 2,6 lần tại CPT(80%) (Arena-only: 2,4 và 1,8 lần);
GSM8K 2,4 và 1,5 lần (Arena-only: 2,0 và 1,2 lần); MMLU 2,1 và 1,3 lần (Arena-only: 2,3 và 1,3 lần).

So sánh trên tập câu chung (hai thí nghiệm loại số câu nhiễm chéo khác nhau; CI 95% bằng bootstrap
2.000 lần trên câu hỏi):

| Benchmark | Số câu chung | APGR Arena | APGR Arena + judge | Hiệu (CI 95%) | Kết luận |
| --- | --- | --- | --- | --- | --- |
| MT Bench | 72 | 0,645 | 0,789 | +0,144 [−0,011, +0,278] | tăng, nhưng CI còn chạm 0 vì chỉ 72 câu |
| MMLU | 13.765 | 0,548 | 0,510 | −0,038 [−0,061, −0,014] | giảm có ý nghĩa |
| GSM8K | 1.300 | 0,507 | 0,574 | +0,067 [+0,014, +0,121] | tăng có ý nghĩa |

Phân bố điểm của router:

| | Arena | Arena + judge |
| --- | --- | --- |
| `P(strong)` trên benchmark | 0,695–0,701 | 0,213–0,229 |
| Độ lệch chuẩn | ≈ 0,0009 | 0,002–0,003 |
| Ngưỡng α tại 20% / 50% / 80% | 0,6986 / 0,6977 / 0,6969 | 0,2222 / 0,2193 / 0,2162 |

Tương quan hạng (Spearman) giữa điểm của hai router chỉ 0,12–0,21: dữ liệu judge đổi gần như toàn bộ
thứ hạng câu hỏi, không chỉ tinh chỉnh.

#### Nhận xét

1. Dữ liệu judge cải thiện rõ trên MT Bench và GSM8K, và hai kết quả này sát với dòng "Arena + judge"
   của bài báo (0,792 so với 0,759; 0,574 so với 0,568) dù dùng embedder khác.
2. Trên MMLU kết quả đi ngược lại: Arena + judge gần random và kém Arena-only. Chưa xác định được nguyên
   nhân. Giả thuyết cần kiểm tra: tín hiệu của bản Arena-only trên MMLU có thể một phần đến từ độ dài
   prompt (tương quan giữa điểm và log độ dài là +0,18), và tín hiệu đó mất đi khi dữ liệu judge chi phối.
3. `P(strong)` vẫn rất hẹp. Khoảng cách giữa các ngưỡng α tăng khoảng 3,6 lần nhưng vẫn chỉ cỡ 0,003,
   nên vấn đề trọng số gần phẳng với `gamma = 10` còn nguyên.
4. Một điểm số chỉ dùng `log(độ dài prompt)` cho APGR 0,552 (MT Bench), 0,581 (MMLU), 0,598 (GSM8K) trên
   tập câu chung. SW Ranking với Arena + judge vượt mốc này trên MT Bench nhưng vẫn thấp hơn trên MMLU và
   GSM8K. Đây là kiểm tra nhanh ngoài CLI, chưa phải baseline chính thức.

#### Hạn chế

- Embedder là `bge-small-en-v1.5`, bài báo dùng `text-embedding-3-small`; chưa phải tái hiện trực tiếp.
- MT Bench chỉ còn 72–73 câu, khoảng tin cậy rộng (APGR Arena + judge: [0,65; 0,96]).
- Tập calibrate lấy 5% từ dữ liệu gộp nên khoảng hai phần ba là prompt judge (theo tỉ lệ hai nguồn);
  repo gốc calibrate trên Arena.
- Ngưỡng nhiễm chéo 0,95 với bge-small loại nhiều câu MMLU hơn repo gốc (276 so với 5), chưa hiệu chỉnh
  theo embedder.
- Phương án đối chứng "tier chỉ tính từ Arena" chưa chạy.

### 2026-11-11 — Cài đặt SW Ranking và chạy với dữ liệu Arena

Thí nghiệm `sw_ranking_toy` (kiểm tra pipeline) và `sw_ranking_arena`. Mục này viết bù ngày 2026-11-15 từ
ADR 0001 và 0003, `docs/experiments/sw_ranking.md`, các dòng ngày 2026-11-11 của bảng kết quả,
`experiments/sw_ranking_arena/`, `experiments/sw_ranking_toy/` và commit `b483452`.

#### Mục tiêu

Có router thật đầu tiên trên khung vừa dựng. Chọn SW Ranking vì không có tham số huấn luyện, chạy được
trên máy không có GPU lớn, và dùng để kiểm chứng toàn bộ pipeline dữ liệu, embedding, tier, hiệu chỉnh
ngưỡng, đánh giá (ADR 0001). Mốc đối chiếu: Bảng 1–3 bài báo RouteLLM, dòng SW Ranking với dữ liệu Arena.

#### Thay đổi trong code

- `src/llmrouter/routers/sw_ranking/bradley_terry.py`: trọng số theo độ giống (phương trình 9), cộng dồn
  trận thành ma trận thắng theo tier, giải Bradley-Terry cho cả lô câu hỏi bằng thuật toán MM.
- `src/llmrouter/routers/sw_ranking/router.py`: class router, đăng ký `sw_ranking`, lưu state
  (`state.npz`, `tiers.json`).
- `configs/experiments/sw_ranking_toy.yaml`, `configs/experiments/sw_ranking_arena.yaml`,
  `configs/embeddings/bge_small.yaml`.
- `tests/test_sw_ranking.py`: so nghiệm Bradley-Terry với `sklearn.LogisticRegression` (bộ giải mà repo
  gốc dùng), dung sai 1e-4.
- Phần dùng chung thêm trong cùng đợt (tải và chuyển đổi Arena, ba benchmark, embedder
  sentence-transformers): xem [`khung.md`](khung.md), mục 2026-11-11.

Năm chỗ khác bài báo / repo gốc (chi tiết ở `../sw_ranking.md`): embedder `bge-small-en-v1.5` thay cho
`text-embedding-3-small`; giải bằng ma trận thắng 10×10 + MM thay cho một hồi quy logistic mỗi câu hỏi;
bỏ trận cùng tier (khoảng 11 nghìn trong 55 nghìn trận Arena); giữ 5% prompt để calibrate; rating toàn
cục bằng Bradley-Terry MM có prior nhỏ.

#### Dữ liệu

- Arena 55k (`lmarena-ai/arena-human-preference-55k`): 57.477 trận thô, 55.361 sau lọc ≥ 16 ký tự.
- Tách 5% prompt để calibrate (2.474 prompt); router fit trên 52.571 trận.
- Tier (10 tier): `gpt-4-1106-preview` rating 1165, tier 0; `mixtral-8x7b-instruct-v0.1` rating 1026,
  tier 2. Cặp tier của hai model đích có 3.757 trận.
- Toy: 2.000 trận giả lập, 1.800 trận fit sau khi tách 10% để calibrate; benchmark toy 300 câu.

#### Quy trình chạy

Lệnh dưới đây dựng lại từ `config.resolved.json`, `thresholds.json` và cột "Thí nghiệm" của bảng kết quả;
thứ tự là vòng đời chuẩn của CLI.

```
python -m llmrouter prepare-data --config configs/experiments/sw_ranking_arena.yaml
python -m llmrouter fit         --config configs/experiments/sw_ranking_arena.yaml
python -m llmrouter calibrate   --config configs/experiments/sw_ranking_arena.yaml --pct 0.2 --pct 0.5 --pct 0.8
python -m llmrouter evaluate    --config configs/experiments/sw_ranking_arena.yaml
python -m llmrouter evaluate    --config ... --set eval.benchmark_name=mmlu  --set eval.benchmark_path=data/processed/bench_mmlu.parquet
python -m llmrouter evaluate    --config ... --set eval.benchmark_name=gsm8k --set eval.benchmark_path=data/processed/bench_gsm8k.parquet
```

- Siêu tham số: `gamma = 10`, `normalize_by_max = true`, `prior = 0`, `batch_size = 256`, seed 0.
  Embedder: `BAAI/bge-small-en-v1.5`, 384 chiều, `max_seq_length = 512`, lô 64, fp16.
- Chi phí: 0 USD, không gọi API.
- Thời gian chạy của đợt này: không có ghi chép.

#### Sự cố

Không có ghi chép về lần chạy hỏng nào của đợt này.

#### Kết quả

Cặp model: gpt-4-1106-preview / mixtral-8x7b-instruct-v0.1. Random: trung bình ± CI 95% trên 20 seed,
cùng benchmark đã lọc nhiễm chéo (cosine ≥ 0,95).

| Benchmark | Dữ liệu fit | Số câu (loại) | APGR | CPT(50%) | CPT(80%) | Random APGR | Random CPT(50%) / CPT(80%) | Bài báo: APGR, CPT(50%), CPT(80%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy | toy 1.800 trận, embedder `hashing` | 300 (0) | 0,833 | 19,0% | 33,3% | 0,495 ± 0,020 | 48,8% / 79,2% | — |
| MT Bench | Arena | 72 (8) | 0,645 | 41,7% | 54,2% | 0,485 ± 0,021 | 50,5% / 80,6% | 0,610, 37,85%, 58,99% |
| MMLU | Arena | 13.889 (153) | 0,549 | 42,3% | 75,9% | 0,499 ± 0,004 | 49,8% / 80,1% | 0,473, 55,82%, 80,25% |
| GSM8K | Arena | 1.310 (9) | 0,509 | 49,9% | 79,9% | 0,510 ± 0,009 | 48,6% / 79,0% | 0,475, 54,43%, 82,11% |

Quy đổi chi phí theo giá trong `configs/models/gpt4_mixtral.yaml`, so với gửi hết cho model mạnh: MT Bench
tiết kiệm 2,37 lần tại CPT(50%) và 1,83 lần tại CPT(80%); MMLU 2,33 và 1,31 lần; GSM8K 1,98 và 1,25 lần.

Phân bố điểm của router trên benchmark: `P(strong)` nằm trong 0,695–0,701, độ lệch chuẩn ≈ 0,0009. Ngưỡng
α tại 20% / 50% / 80% là 0,6986 / 0,6977 / 0,6969.

#### Nhận xét

1. Pipeline chạy trọn vòng: trên toy, SW Ranking hơn random rất rõ (0,833 so với 0,495).
2. Trên dữ liệu thật, router hơn random rõ ở MT Bench và MMLU nhưng ngang random ở GSM8K. Bài báo cũng
   nhận định Arena có ít câu kiểu GSM8K.
3. `P(strong)` gần như hằng số: ba ngưỡng α chỉ cách nhau khoảng 0,002. Với `gamma = 10` và cosine của
   bge-small dồn vào khoảng hẹp, trọng số các trận gần phẳng, nên mọi câu hỏi nhận gần cùng một ước lượng.
4. Số liệu cao hơn dòng tương ứng của bài báo ở cả ba benchmark, nhưng embedder khác nên không phải so
   sánh trực tiếp.

#### Hạn chế và việc tiếp theo

- Chưa chạy với `text-embedding-3-small` nên chưa tái hiện trực tiếp Bảng 1–3.
- MT Bench chỉ còn 72 câu sau khi lọc nhiễm chéo, nhiễu lớn.
- Ngưỡng nhiễm chéo 0,95 với bge-small loại 153 câu MMLU (repo gốc loại 5).
- Chưa quét `gamma`.
- Bước tiếp theo đã làm: bổ sung dữ liệu judge theo mục 4.1.2 bài báo (mục 2026-11-14 ở trên).
