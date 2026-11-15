# 0004. APGR tính bằng tích phân hình thang trên 0%–100%

- Ngày: 2026-11-11
- Trạng thái: chấp nhận

## Bối cảnh
`.claude/rules/evaluation.md` từng viết "APGR ≈ trung bình PGR tại 10 mức 10%, 20%, …, 100%"
(phương trình 8 bài báo). Tính đúng câu chữ đó thì router ngẫu nhiên ra 0,55 chứ không phải 0,50
như bài báo báo cáo. Khung code (`evaluation/metrics.py::apgr`) đã tính bằng hình thang trên 11 điểm
0%, 10%, …, 100% từ đầu, nhưng chưa đối chiếu được với repo gốc.

## Quyết định
Giữ cách hình thang. Đã đối chiếu `routellm/evals/evaluate.py` của repo gốc (commit `0b64fdaf`):
`apgr_metric` dùng `np.trapz` trên đường cong chất lượng theo tỉ lệ gọi model mạnh từ 0 đến 100%, chia
cho diện tích giữa hai đường model mạnh và model yếu — đúng công thức đang cài. Sửa câu chữ trong
`.claude/rules/evaluation.md` cho khớp.

## Phương án đã cân nhắc
- Trung bình PGR tại 10 mức 10%…100%: đúng câu chữ phương trình 8 nhưng lệch +0,05 so với mọi con số
  trong bài báo, nên không so sánh được.

## Hệ quả
Không số liệu nào phải chạy lại (code không đổi). Khác biệt nhỏ còn lại so với repo gốc: repo gốc lấy
ngưỡng theo phân vị điểm của router (`pd.qcut`), nên khi nhiều câu trùng điểm thì các mức không đúng
10%, 20%, …; ở đây mỗi mức gọi đúng tỉ lệ đó và tách hoà theo thứ tự ổn định.
