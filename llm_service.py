"""Claude API integration (official Anthropic SDK). Only schema metadata is sent."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_discovered_gemini_model: str | None = None
_GEMINI_FALLBACK_MODEL = "gemini-flash-lite-latest"
_TRANSIENT_CODES = (500, 502, 503, 504)
_RETRY_DELAYS = (1.5, 3.0, 6.0)  # seconds between attempts
MAX_QUESTION_CHARS = 1000
_PLACEHOLDER_KEYS = {"", "your_api_key_here", "your-api-key", "changeme"}

SETUP_MESSAGE = (
    "No AI API key found. Copy `.env.example` to `.env` and set either GEMINI_API_KEY "
    "(free key: https://aistudio.google.com/apikey) or ANTHROPIC_API_KEY, then restart the app. "
    "You can still write and run SELECT queries manually."
)

SQL_SYSTEM_PROMPT = """You convert questions into SQLite SELECT queries.
Rules:
- Use SQLite syntax only (e.g. strftime for dates, || for concatenation, LIMIT; no ILIKE, no TOP, no MySQL/PostgreSQL functions).
- Use only the tables and columns listed in the schema. Never invent names.
- Produce exactly one read-only SELECT statement (CTEs allowed). Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA or multiple statements.
- Prefer explicit JOIN ... ON using the listed foreign keys, readable aliases, and ORDER BY / LIMIT when the question implies ranking.
- Reply with only the SQL inside one ```sql code block and nothing else.
- If the question cannot be answered from the schema, reply exactly: CANNOT_ANSWER: <short reason>."""

EXPLAIN_SYSTEM_PROMPT = """You explain SQLite queries to non-experts in plain language.
Write a short summary sentence, then a bullet list covering, where present: tables and joins, filters, grouping and aggregations, sorting and limits.
Stay under 180 words. Do not repeat the SQL."""


class LLMError(Exception):
    """Raised with a user-safe message when the AI step fails."""


def _clean_key(name: str) -> str | None:
    key = (os.getenv(name) or "").strip()
    return None if key.lower() in _PLACEHOLDER_KEYS else key


def get_provider() -> str:
    """'anthropic' or 'gemini'. LLM_PROVIDER wins; otherwise use whichever key is set."""
    choice = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if choice in ("anthropic", "gemini"):
        return choice
    if not _clean_key("ANTHROPIC_API_KEY") and _clean_key("GEMINI_API_KEY"):
        return "gemini"
    return "anthropic"


def get_api_key() -> str | None:
    return _clean_key("GEMINI_API_KEY" if get_provider() == "gemini" else "ANTHROPIC_API_KEY")


def is_configured() -> bool:
    return get_api_key() is not None


def get_model() -> str:
    if get_provider() == "gemini":
        return os.getenv("GEMINI_MODEL", "").strip() or _discovered_gemini_model or DEFAULT_GEMINI_MODEL
    return os.getenv("CLAUDE_MODEL", "").strip() or DEFAULT_MODEL


def provider_label() -> str:
    return f"{'Gemini' if get_provider() == 'gemini' else 'Claude'} ({get_model()})"


def _complete(system: str, user: str, max_tokens: int = 1024) -> str:
    if get_api_key() is None:
        raise LLMError(SETUP_MESSAGE)
    if get_provider() == "gemini":
        return _complete_gemini(system, user, max_tokens)
    return _complete_anthropic(system, user, max_tokens)


def pick_gemini_model(models: list[dict]) -> str | None:
    """Choose the newest general-purpose Flash model from a ListModels response."""
    skip = ("lite", "image", "tts", "live", "audio", "embed", "thinking", "exp", "preview", "8b", "vision", "robotics")
    best: tuple[float, int, str] | None = None
    for m in models:
        name = str(m.get("name", "")).removeprefix("models/")
        if "flash" not in name or "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        if any(w in name for w in skip) or name.endswith("-latest"):
            continue
        ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        key = (float(ver.group(1)) if ver else 0.0, -len(name), name)
        if best is None or key > best:
            best = key
    return best[2] if best else None


def _gemini_call_once(model: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(
        f"{_GEMINI_BASE}/models/{model}:generateContent",
        data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _gemini_call(model: str, body: dict, key: str) -> dict:
    """Call Gemini, retrying temporary server errors (500/502/503/504) with backoff."""
    for delay in _RETRY_DELAYS:
        try:
            return _gemini_call_once(model, body, key)
        except urllib.error.HTTPError as exc:
            if exc.code not in _TRANSIENT_CODES:
                raise
            time.sleep(delay)
    return _gemini_call_once(model, body, key)


def _discover_gemini_model(key: str) -> str | None:
    req = urllib.request.Request(f"{_GEMINI_BASE}/models?pageSize=200", headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return pick_gemini_model(json.loads(resp.read().decode("utf-8")).get("models", []))
    except (urllib.error.URLError, ValueError, OSError, TimeoutError):
        return None


def _complete_gemini(system: str, user: str, max_tokens: int) -> str:
    global _discovered_gemini_model
    key = get_api_key() or ""
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": max(max_tokens, 4096)},
    }
    try:
        try:
            data = _gemini_call(get_model(), body, key)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                found = _discover_gemini_model(key)  # configured/default model name is retired: look one up
                if not found or found == get_model():
                    raise
                _discovered_gemini_model = found
                data = _gemini_call(found, body, key)
            elif exc.code in _TRANSIENT_CODES and get_model() != _GEMINI_FALLBACK_MODEL:
                data = _gemini_call(_GEMINI_FALLBACK_MODEL, body, key)  # busy model: try the lighter one
            else:
                raise
    except urllib.error.HTTPError as exc:
        code = exc.code
        if code in (400, 401, 403):
            raise LLMError("Gemini rejected the request. Check that GEMINI_API_KEY is valid and the model name is correct.") from exc
        if code == 404:
            raise LLMError(f"Model '{get_model()}' was not found and no replacement could be found. "
                           "Set GEMINI_MODEL in .env to a model listed at https://aistudio.google.com.") from exc
        if code == 429:
            raise LLMError("Gemini rate limit or free-tier quota reached. Wait a moment and try again.") from exc
        if code in _TRANSIENT_CODES:
            raise LLMError("Google's Gemini servers are overloaded right now (HTTP "
                           f"{code}). The app already retried; wait a minute and click again.") from exc
        raise LLMError(f"The Gemini API returned an error (HTTP {code}). Try again shortly.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError("Could not reach the Gemini API. Check your internet connection.") from exc
    except (ValueError, OSError) as exc:
        raise LLMError("The Gemini API returned an unreadable response.") from exc

    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        text = ""
    if not text:
        raise LLMError("Gemini returned no text (the request may have been blocked). Try rephrasing your question.")
    return text


def _complete_anthropic(system: str, user: str, max_tokens: int) -> str:
    key = get_api_key()
    try:
        import anthropic
    except ImportError as exc:
        raise LLMError("The `anthropic` package is not installed. Run `pip install -r requirements.txt`.") from exc

    try:
        client = anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=2)
        response = client.messages.create(
            model=get_model(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError as exc:
        raise LLMError("The Anthropic API rejected the API key. Check ANTHROPIC_API_KEY in your .env file.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LLMError("The API key does not have permission to use this model. Check CLAUDE_MODEL.") from exc
    except anthropic.NotFoundError as exc:
        raise LLMError(f"Model '{get_model()}' was not found. Set CLAUDE_MODEL in .env to a valid model.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("Rate limit reached. Wait a moment and try again.") from exc
    except anthropic.APITimeoutError as exc:
        raise LLMError("The request to the Anthropic API timed out. Try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Could not reach the Anthropic API. Check your internet connection.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"The Anthropic API returned an error (HTTP {exc.status_code}). Try again shortly.") from exc
    except anthropic.AnthropicError as exc:
        raise LLMError("The Anthropic API request failed.") from exc

    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise LLMError("The model returned an empty response. Try rephrasing your question.")
    return text


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (code fence or bare text)."""
    stripped = text.strip()
    if stripped.upper().startswith("CANNOT_ANSWER"):
        reason = stripped.split(":", 1)[1].strip() if ":" in stripped else "the schema does not contain that information"
        raise LLMError(f"This question can't be answered from the database: {reason}")
    match = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    sql = (match.group(1) if match else stripped).strip()
    if not sql:
        raise LLMError("The model did not return any SQL. Try rephrasing your question.")
    return sql


def generate_sql(question: str, schema_text: str) -> str:
    question = (question or "").strip()
    if not question:
        raise LLMError("Enter a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise LLMError(f"Question is too long (limit {MAX_QUESTION_CHARS} characters).")
    user = f"<schema>\n{schema_text}\n</schema>\n\n<question>\n{question}\n</question>"
    return extract_sql(_complete(SQL_SYSTEM_PROMPT, user))


def fix_sql(question: str, schema_text: str, failed_sql: str, error: str) -> str:
    user = (
        f"<schema>\n{schema_text}\n</schema>\n\n"
        f"<question>\n{(question or 'Not provided').strip()[:MAX_QUESTION_CHARS]}\n</question>\n\n"
        f"<failed_sql>\n{failed_sql}\n</failed_sql>\n\n"
        f"<error>\n{error[:500]}\n</error>\n\n"
        "Return a corrected SQLite SELECT query that fixes the error."
    )
    return extract_sql(_complete(SQL_SYSTEM_PROMPT, user))


def explain_sql(sql: str, schema_text: str) -> str:
    user = f"<schema>\n{schema_text}\n</schema>\n\n<sql>\n{sql}\n</sql>"
    return _complete(EXPLAIN_SYSTEM_PROMPT, user, max_tokens=600)
