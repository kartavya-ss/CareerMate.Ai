"""
Tests for small pure-logic helpers in backend.py.

Note: these test the helper functions only, not the LangGraph agents
themselves (which call out to the Groq LLM and a live Postgres connection,
so they're exercised via manual end-to-end testing instead - see README).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_helpers():
    # Imported lazily inside a function. Setting CAREERMATE_SKIP_DB=1 tells
    # backend.py to skip opening a real Postgres connection at import time,
    # so these pure-logic tests can run without a live database.
    import os

    os.environ.setdefault("GROQ_API_KEY", "test-key")
    os.environ["CAREERMATE_SKIP_DB"] = "1"

    import backend

    return backend


def test_trim_returns_short_text_unchanged():
    backend = _get_helpers()
    text = "short text"
    assert backend._trim(text, limit=100) == text


def test_trim_cuts_long_text_and_flags_it():
    backend = _get_helpers()
    text = "a" * 2000
    trimmed = backend._trim(text, limit=50)
    assert len(trimmed) < len(text)
    assert trimmed.startswith("a" * 50)
    assert "[trimmed for length]" in trimmed


def test_trim_handles_none_and_empty_input():
    backend = _get_helpers()
    assert backend._trim(None) == ""
    assert backend._trim("") == ""


def test_json_from_llm_extracts_object_from_surrounding_text():
    backend = _get_helpers()
    raw = 'Sure, here is the result:\n{"allowed": true, "reason": ""}\nHope that helps!'
    parsed = backend._json_from_llm(raw)
    assert parsed == {"allowed": True, "reason": ""}


def test_json_from_llm_raises_on_no_json():
    backend = _get_helpers()
    try:
        backend._json_from_llm("no json here at all")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_empty_constraints_shape():
    backend = _get_helpers()
    constraints = backend._empty_constraints()
    assert set(constraints.keys()) == {
        "target_role",
        "location",
        "experience_level",
        "must_have_skills",
    }
    assert constraints["must_have_skills"] == []