import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from _analysis import cached_call


def test_cache_is_session_local_and_results_cannot_mutate_it(monkeypatch):
    import _analysis

    calls = []

    def calculate(payload):
        calls.append(payload)
        return {"values": [payload["amount"]]}

    state = {}
    monkeypatch.setattr(_analysis.st, "session_state", state)
    first = cached_call(calculate, {"amount": 5})
    first["values"].append(99)
    assert cached_call(calculate, {"amount": 5}) == {"values": [5]}
    assert len(calls) == 1
    monkeypatch.setattr(_analysis.st, "session_state", {})
    cached_call(calculate, {"amount": 5})
    assert len(calls) == 2


def test_cache_includes_all_inputs_and_is_bounded(monkeypatch):
    import _analysis

    state = {}
    monkeypatch.setattr(_analysis.st, "session_state", state)
    for n in range(12):
        assert cached_call(abs, -n) == n
    assert len(state["_analysis_results"]) == 8


def test_nonfinite_cache_inputs_are_rejected(monkeypatch):
    import _analysis

    monkeypatch.setattr(_analysis.st, "session_state", {})
    with pytest.raises(ValueError):
        cached_call(abs, float("nan"))
