# 0005. Cho phép tier của model yếu xếp trên model mạnh khi gộp dữ liệu judge

- Ngày: 2026-11-14
- Trạng thái: chấp nhận

## Bối cảnh
Để tái hiện dòng "Arena + judge" của bài báo cần gộp `routellm/gpt4_judge_battles` vào dữ liệu fit.
Nguồn này có 109.101 trận, tất cả là `gpt-4-1106-preview` (model_a) đấu `mixtral-8x7b-instruct-v0.1`
(model_b), nhãn do GPT-4 làm judge: GPT-4 thắng 10.136 (9,3%), hoà 24.237 (22,2%), Mixtral "thắng"
74.728 (68,5%). Nhãn mang nghĩa "câu trả lời của model yếu đủ tốt", không phải so kè trực tiếp như Arena.

Rating toàn cục tính trên dữ liệu gộp (đúng như repo gốc, `routers.py` dòng 147–171, commit `0b64fdaf`)
vì thế đảo chiều: Mixtral lên 1148 (tier 0, một mình), GPT-4-1106 xuống 922 (tier 2). Với Arena riêng
thì GPT-4-1106 là 1165 (tier 0) và Mixtral 1026 (tier 2).

`data/tiers.py::check_pair` báo lỗi khi tier model mạnh ≥ tier model yếu, nên `prepare_data` dừng ngay
và không thí nghiệm Arena + judge nào chạy được. Router `sw_ranking` thì chỉ cần hai tier khác nhau:
`P(strong) = p_s / (p_s + p_w)` không phụ thuộc tier nào mang số nhỏ hơn.

## Quyết định
`check_pair` chỉ báo lỗi khi hai model **cùng tier** (hoặc không có tier). Khi model yếu ở tier tốt hơn,
ghi `log.warning` nêu hai tier rồi trả về bình thường. Tier vẫn tính trên toàn bộ nguồn trong
`data.sources`, như repo gốc (ADR 0003: RouteLLM là nền tham chiếu).

## Phương án đã cân nhắc
- Tính rating/tier chỉ từ Arena (thêm tuỳ chọn kiểu `data.tier_sources`): giữ được thứ bậc "thật" của các
  model, nhưng khác repo gốc nên không còn là tái hiện. Để lại làm thí nghiệm đối chứng sau.
- Giữ nguyên kiểm tra, đổi nhãn judge cho "hợp" thứ tự: sửa dữ liệu để vừa code, không chấp nhận được.

## Hệ quả
Không số liệu nào phải chạy lại: kiểm tra chỉ lỏng hơn, tier và kết quả của các thí nghiệm Arena-only
không đổi. Với dữ liệu gộp, `P(strong)` sẽ nằm dưới 0,5 ở phần lớn câu hỏi (nhãn judge nghiêng về model
yếu); điều này không ảnh hưởng APGR/CPT vì chúng chỉ phụ thuộc thứ hạng, và ngưỡng α vẫn đến từ
`calibrate`. Tên "tier 0 = tốt nhất" từ nay chỉ đúng theo nhãn của dữ liệu fit, không còn là thứ bậc
năng lực khi có nguồn judge.
