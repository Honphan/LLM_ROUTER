# 0001. Làm SW Ranking trước các router có tham số

- Ngày: 2026-11-08
- Trạng thái: chấp nhận

## Bối cảnh
Cần một router chạy được trên máy 8GB RAM không GPU để dựng và kiểm chứng toàn bộ pipeline.

## Quyết định
Cài SW Ranking đầu tiên: không có tham số huấn luyện, giải Bradley-Terry lúc suy luận. Pipeline dữ liệu,
embedding, tier, hiệu chỉnh ngưỡng và đánh giá APGR/CPT dựng cho nó được các router sau dùng lại.

## Phương án đã cân nhắc
- Matrix Factorization trước: kết quả bài báo tốt hơn, nhưng cần vòng huấn luyện và GPU để lặp nhanh.

## Hệ quả
Phần dùng chung được kiểm bằng một router đơn giản; MF/BERT chỉ cần thêm `fit()` và `strong_win_rate()`.
