"""The context-notes advisor.

The design contract worth protecting here is that this is *local first*: with
no API key and no network it must still produce useful, relevant insights, and
it must never leak exact figures off the machine.
"""
from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import pytest

# Every provider the module knows about. A key left in the environment (or
# loaded from .env) would make "no key configured" tests silently pass.
PROVIDER_KEYS = ("FINREC_LLM_API_KEY", "OPENAI_API_KEY",
                 "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _clear_keys(monkeypatch):
    for name in PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)


from finrec.advisor_llm import (
    Insight,
    llm_available,
    llm_status,
    local_insights,
    redacted_payload,
    tailored_advice,
)
from finrec.profile import Profile


def _profile(**kw) -> Profile:
    base = dict(age=38, salary=180_000, state="CA",
                cash=60_000, taxable_investments=140_000, traditional_401k=310_000)
    base.update(kw)
    return Profile(**base)


class TestLocalInsights:
    @pytest.mark.parametrize("note,expect", [
        ("we are expecting a baby in the spring", "529"),
        ("my wife is pregnant", "529"),
        ("I get RSUs that vest quarterly", "concentration"),
        ("a big chunk of my comp is rsu", "concentration"),
        ("thinking about starting a business", "income"),
        ("I might get laid off", "cash"),
        ("caring for my aging parents", "care"),
        ("we want to buy a house in two years", "short"),
    ])
    def test_keywords_fire(self, note, expect):
        out = local_insights(note)
        assert out, f"no insight for {note!r}"

    def test_no_notes_means_no_insights(self):
        assert local_insights("") == []

    def test_unrelated_notes_do_not_invent_advice(self):
        out = local_insights("I enjoy hiking on weekends.")
        assert out == []

    def test_insights_are_structured(self):
        out = local_insights("my wife is pregnant")
        assert all(isinstance(i, Insight) for i in out)
        assert all(i.title and i.detail for i in out)

    def test_prefix_matching_is_not_blocked_by_word_boundaries(self):
        """A trailing \\b after an alternation stops 'pregnan' matching 'pregnant'."""
        for variant in ("pregnant", "pregnancy", "we are pregnant now"):
            assert local_insights(variant), variant


class TestRedaction:
    def test_balances_are_bucketed_not_exact(self):
        payload = redacted_payload(_profile(cash=63_421, taxable_investments=147_802), "note")
        flat = str(payload)
        assert "63,421" not in flat and "63421" not in flat
        assert "147802" not in flat

    def test_income_is_rounded(self):
        payload = redacted_payload(_profile(salary=183_450), "note")
        assert "183450" not in str(payload)

    def test_the_note_itself_is_included(self):
        """The user opts in knowing the note is sent; it is the whole point."""
        payload = redacted_payload(_profile(), "expecting a baby")
        assert "expecting a baby" in str(payload)


class TestGracefulDegradation:
    def test_no_key_means_unavailable(self, monkeypatch):
        _clear_keys(monkeypatch)
        assert llm_available() is False
        assert "key" in llm_status().lower()

    def test_tailored_advice_works_with_no_key(self, monkeypatch):
        _clear_keys(monkeypatch)
        out = tailored_advice(_profile(), "my wife is pregnant", use_llm=True)
        assert out, "must fall back to local rules rather than returning nothing"

    def test_llm_is_never_called_without_consent(self, monkeypatch):
        called = []
        monkeypatch.setattr("finrec.advisor_llm.llm_insights",
                            lambda *a, **k: called.append(1) or [])
        tailored_advice(_profile(), "my wife is pregnant", use_llm=False)
        assert not called


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------

import finrec.advisor_llm as _mod  # noqa: E402


class TestProviderSelection:
    def test_gemini_is_preferred_when_both_keys_exist(self, monkeypatch):
        """Gemini's Flash-Lite tier is the cheaper of the two."""
        _clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setenv("OPENAI_API_KEY", "o-key")
        monkeypatch.delenv("FINREC_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("FINREC_LLM_MODEL", raising=False)

        name, key, base, model = _mod._provider()
        assert name == "Gemini"
        assert key == "g-key"
        assert "generativelanguage.googleapis.com" in base
        assert "flash-lite" in model

    def test_openai_is_used_when_it_is_the_only_key(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "o-key")
        monkeypatch.delenv("FINREC_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("FINREC_LLM_MODEL", raising=False)

        name, key, base, model = _mod._provider()
        assert (name, key) == ("OpenAI", "o-key")
        assert "api.openai.com" in base

    def test_no_provider_without_any_key(self, monkeypatch):
        _clear_keys(monkeypatch)
        assert _mod._provider() is None
        assert _mod.provider_name() is None

    def test_an_explicit_model_overrides_the_default(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setenv("FINREC_LLM_MODEL", "something-else")
        assert _mod._provider()[3] == "something-else"

    def test_status_names_the_provider_and_model(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.delenv("FINREC_LLM_MODEL", raising=False)
        status = _mod.llm_status()
        assert "Gemini" in status and "flash-lite" in status

    def test_status_tells_you_which_variable_to_set(self, monkeypatch):
        _clear_keys(monkeypatch)
        assert "GEMINI_API_KEY" in _mod.llm_status()

    def test_the_default_model_is_an_unpinned_alias(self):
        """Pinned Gemini model names get retired and start returning 404.

        Both names this was first written against (gemini-2.0-flash-lite and
        gemini-2.5-flash-lite) are already gone, which would have silently
        disabled the feature for everyone.
        """
        assert _mod.GEMINI_MODEL == "gemini-flash-lite-latest"
        assert not re.search(r"\d+\.\d+", _mod.GEMINI_MODEL)


class TestDotEnv:
    def test_env_file_is_read(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('GEMINI_API_KEY="from-file"\n# a comment\nOTHER=x\n')
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(_mod, "__file__", str(tmp_path / "pkg" / "advisor_llm.py"))
        (tmp_path / "pkg").mkdir(exist_ok=True)
        _mod._load_dotenv()
        assert os.environ.get("GEMINI_API_KEY") == "from-file"
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def test_a_real_environment_variable_wins(self, tmp_path, monkeypatch):
        """Otherwise a stale file would override a deliberate override."""
        env = tmp_path / ".env"
        env.write_text("GEMINI_API_KEY=from-file\n")
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        monkeypatch.setattr(_mod, "__file__", str(tmp_path / "pkg" / "advisor_llm.py"))
        (tmp_path / "pkg").mkdir(exist_ok=True)
        _mod._load_dotenv()
        assert os.environ["GEMINI_API_KEY"] == "from-env"

    def test_a_missing_env_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "__file__", str(tmp_path / "nope" / "advisor_llm.py"))
        _mod._load_dotenv()  # must not raise

    def test_the_key_is_not_committed(self):
        """The one rule that matters for a secret."""
        import subprocess
        root = Path(_mod.__file__).resolve().parents[1]
        result = subprocess.run(["git", "check-ignore", ".env"], cwd=root,
                                capture_output=True, text=True)
        assert result.returncode == 0, ".env is not git-ignored"


class TestRequestShape:
    """What actually goes over the wire, without making a network call."""

    def _capture(self, monkeypatch):
        sent = {}

        class _Response:
            def read(self):
                return json.dumps({"choices": [{"message": {"content": json.dumps(
                    {"insights": [{"title": "T", "detail": "D",
                                   "confidence": "high"}]})}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(request, timeout=None):
            sent["url"] = request.full_url
            sent["headers"] = dict(request.headers)
            sent["body"] = json.loads(request.data)
            return _Response()

        monkeypatch.setattr(_mod.urllib.request, "urlopen", fake_urlopen)
        return sent

    def test_it_calls_the_gemini_endpoint_with_a_bearer_token(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.delenv("FINREC_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("FINREC_LLM_MODEL", raising=False)
        sent = self._capture(monkeypatch)

        out = _mod.llm_insights(Profile(salary=200_000), "expecting a baby", consent=True)

        assert sent["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        assert sent["headers"]["Authorization"] == "Bearer g-key"
        assert sent["body"]["model"] == "gemini-flash-lite-latest"
        assert out and out[0].source == "llm"

    def test_exact_balances_never_leave_the_machine(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        sent = self._capture(monkeypatch)

        profile = Profile(salary=263_517, taxable_investments=412_345,
                          cash=88_213)
        _mod.llm_insights(profile, "some notes", consent=True)

        wire = json.dumps(sent["body"])
        for exact in ("263517", "412345", "88213", "263,517"):
            assert exact not in wire, f"{exact} was sent verbatim"

    def test_a_retired_model_reports_the_real_reason(self, monkeypatch):
        """A 404 here means the model name died, not that the key is wrong."""
        _clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")

        def fake_urlopen(request, timeout=None):
            raise _mod.urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {},
                io.BytesIO(json.dumps({"error": {
                    "message": "This model is no longer available."}}).encode()))

        monkeypatch.setattr(_mod.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError) as exc:
            _mod.llm_insights(Profile(salary=200_000), "notes", consent=True)
        assert "no longer available" in str(exc.value)
        assert "API key" not in str(exc.value), "misleading cause"
