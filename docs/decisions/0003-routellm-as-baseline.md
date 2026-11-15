# 0003. RouteLLM là nền tham chiếu, không phải mục tiêu cuối

- Ngày: 2026-11-11
- Trạng thái: chấp nhận

## Bối cảnh
Repo chỉ mới có cấu hình Claude Code và tài liệu, chưa có code. Đóng góp của luận văn là các thuật toán
định tuyến tự phát triển; bài báo RouteLLM (ICLR 2025) và repo `lm-sys/RouteLLM` chỉ là điểm tựa.

## Quyết định
- Viết khung `llmrouter` từ đầu theo giao diện trong `CLAUDE.md`; không fork repo gốc.
- Repo gốc đặt ở `third_party/RouteLLM` (chỉ đọc, ngoài git) để đối chiếu quy ước tier/Elo, APGR/CPT
  và định dạng benchmark.
- Làm theo thứ tự: cấu trúc cơ bản chạy end-to-end offline với router `random` → SW Ranking làm router
  tham chiếu, đối chiếu với repo gốc → dữ liệu thật → thuật toán tự phát triển.
- Router của bài báo cần GPU lớn (BERT, Causal LLM) chỉ trích số liệu từ bài báo khi so sánh.

## Phương án đã cân nhắc
- Fork repo gốc rồi sửa: nhanh có số liệu, nhưng giao diện router, cấu hình và đánh giá của repo gốc
  không tách theo các bất biến của dự án, khó cắm thuật toán mới và khó kiểm thử offline.

## Hệ quả
Khung phải được kiểm chứng độc lập (ví dụ tính tay, random ≈ 0.5) trước khi có router thật. Khi cài
SW Ranking cần một test so kết quả với cài đặt trong `third_party/RouteLLM`.

Cập nhật 2026-11-11: chưa clone repo gốc; đối chiếu bằng cách đọc mã nguồn tại commit
`0b64fdafe049e596a3f5657c219329f24af24198`. Dữ liệu benchmark tải từ đúng commit này
(`configs/data/routellm_public.yaml`). Test của SW Ranking so phần Bradley-Terry với
`sklearn.LogisticRegression`, là bộ giải mà repo gốc dùng.
