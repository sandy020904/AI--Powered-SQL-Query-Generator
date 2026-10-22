# SQL Query Generator Using LLMs, SQLite & Generative AI

Ask questions in plain English; Claude writes SQLite SQL from your schema, you review/edit it, and it runs safely against a read-only database.

## Setup (Windows, PowerShell)
```powershell
cd sql-query-generator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env and set ANTHROPIC_API_KEY
streamlit run app.py
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`.

## Configuration (`.env`)
| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Free key from https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model name (auto-replaced if not found) |
| `LLM_PROVIDER` | auto | `gemini` or `anthropic`; auto-picks whichever key is set |
| `ANTHROPIC_API_KEY` | – | Paid Claude key from https://console.anthropic.com |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Model used for SQL generation/explanations |
| `MAX_RESULT_ROWS` | 1000 | Row cap per query |
| `QUERY_TIMEOUT_SECONDS` | 10 | Per-query time limit |
| `MAX_UPLOAD_MB` | 50 | Upload size limit |

## Tests
```powershell
python -m pytest
```

## Security layers
1. `sqlglot` parses every query: one statement, SELECT only, blocked functions, no `sqlite_*` tables, no unknown tables.
2. Connections open with `mode=ro`, plus a SQLite authorizer that denies everything except reads.
3. Progress-handler timeout and row cap.
4. Database names are whitelisted and resolved inside `data/`; uploads are header-checked and size-limited.
5. Only schema metadata (never row data) is sent to Claude. The API key is never displayed or logged.

## Files
`app.py` UI · `database.py` SQLite layer + sample data · `llm_service.py` Claude calls · `sql_validator.py` SQL safety · `tests/` pytest suite
