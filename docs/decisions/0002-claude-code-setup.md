# 0002. Cấu trúc Claude Code cho dự án

- Ngày: 2026-11-08
- Trạng thái: chấp nhận

## Bối cảnh
Claude Code sẽ cài nhiều router lần lượt. Cần nó làm đúng quy ước mà không phải nhắc lại mỗi phiên,
không phá phần dùng chung, không sửa kết quả thí nghiệm và không tiêu tiền API ngoài ý muốn.

## Quyết định
Chia hướng dẫn theo mức độ ràng buộc:
- Luôn đúng, áp dụng mọi nơi → `CLAUDE.md` (dưới ~150 dòng).
- Chỉ đúng với một phần code → `.claude/rules/*.md` có `paths:`.
- Quy trình nhiều bước → `.claude/skills/`; bước tốn tiền đặt `disable-model-invocation: true`.
- Việc nặng ngữ cảnh (review, phân tích) → `.claude/agents/`.
- Luật không có ngoại lệ → `permissions.deny` và hooks trong `.claude/settings.json`.

## Phương án đã cân nhắc
- Một CLAUDE.md dài chứa tất cả: tốn ngữ cảnh mọi phiên và giảm mức tuân thủ.

## Hệ quả
Thêm router mới không cần sửa cấu hình Claude; chỉ thêm rule/skill khi xuất hiện quy ước mới.
