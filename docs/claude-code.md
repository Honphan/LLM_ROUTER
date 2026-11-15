# Làm việc với Claude Code trong llm-router

Tài liệu cho người. Claude đọc CLAUDE.md và `.claude/`, không cần file này.

## Mỗi thứ nằm ở đâu

| Bạn muốn | Sửa | Nạp khi nào |
| --- | --- | --- |
| Quy ước đúng ở mọi nơi | `CLAUDE.md` | mọi phiên |
| Trạng thái router, việc đang làm | `docs/ROADMAP.md` (CLAUDE.md import) | mọi phiên |
| Quy ước cho một phần code | `.claude/rules/<chủ đề>.md` với `paths:` | khi Claude đọc file khớp |
| Quy trình nhiều bước | `.claude/skills/<tên>/SKILL.md` | khi gọi `/<tên>` hoặc Claude thấy liên quan |
| Trợ lý chuyên biệt, ngữ cảnh riêng | `.claude/agents/<tên>.md` | khi được giao việc |
| Luật bắt buộc | `.claude/settings.json` (`permissions`, `hooks`) | luôn được Claude Code thực thi |
| Quyền / thông báo cá nhân | `.claude/settings.local.json` (mẫu: `.example`) | chỉ máy bạn |
| Ghi chú cá nhân cho Claude | `CLAUDE.local.md` | chỉ máy bạn |

## Skills có sẵn

| Lệnh | Việc | Claude tự gọi? |
| --- | --- | --- |
| `/add-router <tên>` | cài router mới: kế hoạch → test toán → router → toy → tài liệu | có |
| `/run-experiment <config>` | fit → calibrate → evaluate, ghi log riêng + log chung | không (có thể tốn tiền) |
| `/compare-experiments` | bảng so sánh với nhau và với bài báo | có |
| `/analyze-misroutes <thí nghiệm>` | phân tích câu định tuyến sai trong subagent | có |
| `/add-data-source <nguồn>` | nạp nguồn battle hoặc benchmark mới | có |
| `/label-domain-data <file>` | sinh câu trả lời + judge cho dữ liệu trong miền | không (tốn tiền) |
| `/log-decision <tiêu đề>` | ghi ADR | có |

Subagents: `code-reviewer` (rà diff theo bất biến), `experiment-analyst` (đọc kết quả),
`method-auditor` (đối chiếu code với bài báo).

## Hooks

| Sự kiện | Script | Làm gì |
| --- | --- | --- |
| SessionStart | `session_context.py` | nhánh, số file chưa commit, thiếu .venv/.env, 3 thí nghiệm gần nhất |
| PreToolUse Edit/Write | `protect_paths.py` | chặn sửa `.env`, `data/raw/`, `experiments/`, `third_party/`, `.venv/` |
| PreToolUse Bash | `guard_bash.py` | chặn `pip install` lẻ, xoá đệ quy thư mục chính, in `.env`; hỏi trước `git push -f`, `reset --hard` |
| PostToolUse Edit/Write | `format_python.py` | `ruff format` + `ruff check --fix` file .py vừa sửa; lỗi còn lại trả cho Claude |
| Stop | `stop_checks.py` | nếu .py đổi: ruff + `pytest -m "not slow and not network"`; đỏ thì Claude phải sửa tiếp. Nếu file chưa commit trong `src/`, `configs/`, `tests/` mới hơn mọi file log (`docs/experiments/LOG.md`, `docs/experiments/log/*.md`): chặn một lần, nhắc Claude ghi log |

Hooks viết bằng Python thư viện chuẩn, chạy bằng `python3` hệ thống, tự dùng `ruff`/`python` trong
`.venv` nếu có. Trên Windows dùng WSL2. Tắt tạm mọi hook một lần chạy:
`claude --settings '{"disableAllHooks": true}'`.

## Mở rộng

- **Router mới**: không cần sửa `.claude/`; chạy `/add-router <tên>`.
- **Quy ước mới cho một thư mục**: thêm file trong `.claude/rules/` với `paths:`.
- **Lặp lại cùng một chuỗi bước lần thứ ba**: biến thành skill.
- **Claude vi phạm cùng một luật hai lần**: chuyển luật đó từ văn bản sang hook hoặc `permissions.deny`.
- Kiểm tra cấu hình đã nạp: `/context` (CLAUDE.md, rules), `/hooks`, `/permissions`, `/skills`.
  Rà mâu thuẫn trong hướng dẫn: `/doctor prompt-audit`.

## Bổ sung vào .gitignore

```gitignore
.env
.env.local
CLAUDE.local.md
.claude/settings.local.json
.claude/.state/
.claude/agent-memory-local/
```
