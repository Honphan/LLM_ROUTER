# SW Ranking (similarity-weighted ranking)

Router tham chiếu đầu tiên, theo mục 4.2 bài báo RouteLLM (phương trình 9–10). Code:
`src/llmrouter/routers/sw_ranking/`.

## Trực giác

Không huấn luyện gì trước. Khi có câu hỏi mới, router tìm xem trong dữ liệu Arena những câu **giống**
câu này thì model mạnh thắng model yếu thường xuyên tới mức nào:

1. Embed câu hỏi, tính cosine với prompt của mọi trận đã lưu.
2. Trận càng giống thì trọng số càng lớn.
3. Gộp các trận (đã đổi tên model thành tier) thành bảng thắng–thua có trọng số giữa các tier, rồi
   fit Bradley-Terry để ra hệ số mỗi tier.
4. `P(strong) = σ(ξ_strong − ξ_weak)`. Giá trị này so với ngưỡng α từ `calibrate`.

Dùng tier (10 nhóm model theo Elo) thay cho từng model để có đủ trận cho mỗi cặp: Arena có rất ít trận
trực tiếp giữa đúng hai model mạnh/yếu.

## Công thức ↔ code

| Bài báo | Ý nghĩa | Code |
| --- | --- | --- |
| pt 9: `ω_i = γ^(1 + S(q, q_i))` | trọng số trận i với câu q; `S` là cosine chia cho cosine lớn nhất | `bradley_terry.similarity_weights` |
| pt 10: `argmin_ξ Σ ω_i · ℓ(l_i, σ(ξ_wi − ξ_li))` | Bradley-Terry có trọng số, hoà = nửa thắng mỗi bên | `bradley_terry.aggregate_tier_wins` + `fit_bradley_terry_batch` |
| `P(win_s \| q) = σ(ξ_s − ξ_w)` | xác suất model mạnh thắng | `bradley_terry.win_probability` |
| phụ lục A: tier theo Elo, chia đoạn tối thiểu phương sai | 10 tier | `data/tiers.py` (dùng chung) |

## Khác bài báo / repo gốc

1. **Embedder.** Bài báo dùng `text-embedding-3-small` (OpenAI). Thí nghiệm `sw_ranking_arena` dùng
   `BAAI/bge-small-en-v1.5` chạy local để không tốn tiền, nên số liệu không so trực tiếp được với
   Bảng 1–3. Phân bố cosine của hai embedder khác nhau, mà trọng số phụ thuộc trực tiếp vào cosine.
2. **Cách giải.** Repo gốc chạy một `LogisticRegression` trên toàn bộ trận cho mỗi câu hỏi. Ở đây các
   trận được cộng dồn thành ma trận thắng 10×10 theo tier rồi giải Bradley-Terry cho cả lô câu hỏi
   bằng thuật toán MM. Cùng một nghiệm tối ưu (test so với sklearn, dung sai 1e-4), nhanh hơn nhiều.
3. **Trận cùng tier bị bỏ.** Hai model cùng tier không cho thông tin gì trong Bradley-Terry mức tier.
   Repo gốc vẫn giữ các trận này do một chỗ ghi đè chỉ số, làm lệch nhẹ hệ số của tier đó. Trên Arena
   có khoảng 11 nghìn trong 55 nghìn trận thuộc loại này.
4. **Tập calibrate.** 5% prompt Arena được giữ lại để tìm ngưỡng α, nên router fit trên 95% còn lại.
5. **Tier.** Rating toàn cục tính bằng Bradley-Terry MM có prior nhỏ thay vì hồi quy logistic; ranh
   giới tier có thể lệch một model so với repo gốc.

## Dữ liệu judge (Arena + judge)

Thí nghiệm `sw_ranking_arena_judge` gộp thêm `routellm/gpt4_judge_battles` (mục 4.1.2 bài báo): 109.101
trận, tất cả là `gpt-4-1106-preview` đấu `mixtral-8x7b-instruct-v0.1`, nhãn do GPT-4 chấm. Chuyển đổi ở
`data/sources/gpt4_judge.py`, cột `source = gpt4_judge`, trọng số 1.0 như repo gốc.

Nhãn judge khác nghĩa nhãn Arena: Mixtral "thắng" khi câu trả lời của nó được chấm là đủ tốt, và điều
đó xảy ra ở 68,5% số trận (GPT-4 thắng 9,3%, hoà 22,2%). Hai hệ quả:

- Rating toàn cục trên dữ liệu gộp xếp Mixtral lên tier 0 (một mình) và GPT-4-1106 xuống tier 2. Repo gốc
  cũng tính tier trên dữ liệu gộp; ở đây `check_pair` chỉ cấm cùng tier (ADR 0005).
- `P(strong)` phần lớn nằm dưới 0,5. Nó vẫn là điểm để xếp hạng câu hỏi; ngưỡng α đến từ `calibrate`
  nên quy tắc `P(strong) ≥ α → model mạnh` không đổi.

Cặp tier của hai model đích có khoảng 110 nghìn trận (trước đó 3,8 nghìn), nên ước lượng theo từng câu
hỏi dựa trên nhiều trận giống hơn hẳn so với khi chỉ có Arena.

Kết quả và so sánh với bài báo: `docs/experiments/LOG.md`, mục "Ghi chép chi tiết" ngày 2026-11-14.

## Núm vặn (`router.params`)

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `gamma` | 10 | cơ số trọng số; lớn hơn thì trận giống được ưu tiên mạnh hơn. `gamma=1` → mọi trận ngang nhau, mọi câu ra cùng một xác suất |
| `normalize_by_max` | true | chia cosine cho cosine lớn nhất của câu đó (như repo gốc) |
| `prior` | 0.0 | số trận ảo thắng/thua với đối thủ trung bình; 0 = MLE thuần như bài báo |
| `max_iter`, `tol` | 5000, 1e-9 | điều kiện dừng của MM |
| `batch_size` | 256 | số câu xử lý mỗi lô lúc suy luận; đỉnh bộ nhớ khoảng lô × số trận × 20 byte (Arena + judge, 144 nghìn trận: dùng 64) |
| `store_dtype` | float32 | kiểu lưu embedding trong state; `float16` giảm nửa dung lượng |

State lưu trong `experiments/<tên>/router/`: `state.npz` (embedding các trận, phần thắng mỗi bên, nhóm
cặp tier) và `tiers.json`. Không lưu prompt.

## Thí nghiệm gợi ý

- Quét `gamma` (2, 5, 10, 30, 100) và `normalize_by_max`: với embedder có cosine dồn vào khoảng hẹp
  (bge, e5) trọng số gần như phẳng, có thể cần `gamma` lớn hơn nhiều so với bài báo.
- Đổi embedder: `bge-m3` (đa ngữ, cần cho tiếng Việt), `text-embedding-3-small` (để tái hiện bài báo).
- Đối chứng cho dữ liệu judge: tính tier chỉ từ Arena rồi mới gộp trận judge (ADR 0005, phương án bị loại).
- Chỉ dùng top-k trận giống nhất thay vì toàn bộ.
