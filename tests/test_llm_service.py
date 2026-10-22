import pytest

import llm_service as llm


def test_extract_sql_from_fenced_block():
    assert llm.extract_sql("Sure!\n```sql\nSELECT 1;\n```") == "SELECT 1;"


def test_extract_sql_from_plain_text():
    assert llm.extract_sql("  SELECT 2  ") == "SELECT 2"


def test_cannot_answer_raises_with_reason():
    with pytest.raises(llm.LLMError, match="no weather"):
        llm.extract_sql("CANNOT_ANSWER: no weather table")


def test_missing_key_gives_setup_message(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_api_key_here")
    assert not llm.is_configured()
    with pytest.raises(llm.LLMError, match="API key"):
        llm.generate_sql("top customers", "TABLE customers")


def test_provider_auto_selects_gemini_when_only_gemini_key_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_api_key_here")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert llm.get_provider() == "gemini" and llm.is_configured()
    assert llm.get_model() == llm.DEFAULT_GEMINI_MODEL


def test_empty_and_overlong_questions_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(llm.LLMError):
        llm.generate_sql("   ", "schema")
    with pytest.raises(llm.LLMError):
        llm.generate_sql("x" * (llm.MAX_QUESTION_CHARS + 1), "schema")


def test_pick_gemini_model_prefers_newest_plain_flash():
    gc = ["generateContent"]
    models = [
        {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": gc},
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": gc},
        {"name": "models/gemini-2.5-flash-lite", "supportedGenerationMethods": gc},
        {"name": "models/gemini-3-flash-preview", "supportedGenerationMethods": gc},
        {"name": "models/gemini-2.5-flash-image", "supportedGenerationMethods": gc},
        {"name": "models/gemini-flash-latest", "supportedGenerationMethods": gc},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
    ]
    assert llm.pick_gemini_model(models) == "gemini-2.5-flash"
    assert llm.pick_gemini_model([]) is None


def _http_error(code):
    import urllib.error
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def test_gemini_retries_transient_503_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(model, body, key):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(503)
        return {"ok": True}

    monkeypatch.setattr(llm, "_gemini_call_once", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm._gemini_call("m", {}, "k") == {"ok": True}
    assert calls["n"] == 3


def test_gemini_does_not_retry_non_transient_errors(monkeypatch):
    calls = {"n": 0}

    def bad(model, body, key):
        calls["n"] += 1
        raise _http_error(403)

    monkeypatch.setattr(llm, "_gemini_call_once", bad)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(Exception):
        llm._gemini_call("m", {}, "k")
    assert calls["n"] == 1


def test_gemini_falls_back_to_lite_model_and_reports_overload(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-flash-latest")
    seen = []

    def fake(model, body, key):
        seen.append(model)
        if model == llm._GEMINI_FALLBACK_MODEL:
            return {"candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}]}
        raise _http_error(503)

    monkeypatch.setattr(llm, "_gemini_call", fake)
    assert llm._complete("s", "u") == "SELECT 1"
    assert seen == ["gemini-flash-latest", llm._GEMINI_FALLBACK_MODEL]

    def always_503(model, body, key):
        raise _http_error(503)

    monkeypatch.setattr(llm, "_gemini_call", always_503)
    with pytest.raises(llm.LLMError, match="overloaded"):
        llm._complete("s", "u")
