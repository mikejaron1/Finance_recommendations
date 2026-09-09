"""The HTTP API — the seam the eventual website will be built on.

Two things matter here beyond "does it return 200": that saved data comes back
*exactly* as it went in, and that the plan routes are matched before the
catch-all analysis route that would otherwise swallow them.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from finrec.api import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))


@pytest.fixture
def sample() -> dict:
    return {
        "name": "My plan",
        "profile": {
            "salary": 200_000, "bonus": 50_000, "stock_comp": 75_000,
            "location": "Austin, TX", "age": 38, "cash": 60_000,
            "annual_401k_contribution": 23_500, "context_notes": "expecting a baby",
        },
    }


class TestHealth:
    def test_health_reports_the_store(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["store"]["ok"] is True
        assert body["store"]["schema_version"] >= 1

    def test_defaults_lists_the_plan_services(self, client):
        services = client.get("/api/defaults").json()["services"]
        for name in ("plans/list", "plans/save", "plans/restore"):
            assert name in services


class TestPlanRoutes:
    def test_save_then_read_back(self, client, sample):
        saved = client.post("/api/plans", json=sample)
        assert saved.status_code == 200
        assert saved.json()["simple"]["version"] == 1

        got = client.get("/api/plans/my-plan").json()["simple"]["profile"]
        assert got["salary"] == 200_000
        assert got["bonus"] == 50_000
        assert got["stock_comp"] == 75_000
        assert got["context_notes"] == "expecting a baby"

    def test_saving_twice_appends_a_version(self, client, sample):
        client.post("/api/plans", json=sample)
        second = client.post("/api/plans", json=sample)
        assert second.json()["simple"]["version"] == 2

    def test_history_and_restore(self, client, sample):
        client.post("/api/plans", json=sample)
        changed = {**sample, "profile": {**sample["profile"], "salary": 999_000}}
        client.post("/api/plans", json=changed)
        assert client.get("/api/plans/my-plan").json()["simple"]["profile"]["salary"] == 999_000

        versions = client.get("/api/plans/my-plan/history").json()["simple"]["versions"]
        assert [v["version"] for v in versions] == [2, 1]

        client.post("/api/plans/my-plan/restore/1")
        assert client.get("/api/plans/my-plan").json()["simple"]["profile"]["salary"] == 200_000

    def test_reading_an_old_version_does_not_change_the_current_one(self, client, sample):
        client.post("/api/plans", json=sample)
        changed = {**sample, "profile": {**sample["profile"], "salary": 999_000}}
        client.post("/api/plans", json=changed)
        old = client.get("/api/plans/my-plan", params={"version": 1}).json()
        assert old["simple"]["profile"]["salary"] == 200_000
        assert client.get("/api/plans/my-plan").json()["simple"]["profile"]["salary"] == 999_000

    def test_delete_then_missing(self, client, sample):
        client.post("/api/plans", json=sample)
        assert client.delete("/api/plans/my-plan").status_code == 200
        assert client.get("/api/plans/my-plan").status_code == 404

    def test_unknown_plan_is_404(self, client):
        assert client.get("/api/plans/nope").status_code == 404
        assert client.get("/api/plans/nope/history").json()["simple"]["count"] == 0
        assert client.post("/api/plans/nope/restore/1").status_code == 404
        assert client.delete("/api/plans/nope").status_code == 404

    def test_save_requires_a_profile(self, client):
        assert client.post("/api/plans", json={"name": "x"}).status_code == 422

    def test_export_returns_everything(self, client, sample):
        client.post("/api/plans", json=sample)
        dump = client.get("/api/export").json()["simple"]
        assert dump["plans"][0]["profile"]["bonus"] == 50_000
        assert dump["plans"][0]["history"]


class TestUserScoping:
    def test_local_account_matches_trusted_environment_configuration(self, client, sample, monkeypatch):
        from finrec import db, storage
        monkeypatch.setenv("FINREC_USER", "local-fixture@example.com")
        client.post("/api/plans", json=sample)
        assert len(storage.list_plans(user_id=db.default_user_id())) == 1

    def test_query_user_cannot_assert_identity(self, client, sample):
        client.post("/api/plans", json=sample, params={"user": "alice@example.com"})
        assert client.get("/api/plans", params={"user": "bob@example.com"}).json()["simple"]["count"] == 1

    def test_configured_account_not_query_controls_ownership(self, client, sample, monkeypatch):
        monkeypatch.setenv("FINREC_API_TOKEN", "fixture-token")
        monkeypatch.setenv("FINREC_API_USER", "alice@example.com")
        headers = {"Authorization": "Bearer fixture-token"}
        client.post("/api/plans", json=sample, params={"user": "bob@example.com"}, headers=headers)
        assert client.get("/api/plans", headers=headers).json()["simple"]["count"] == 1
        monkeypatch.setenv("FINREC_API_USER", "bob@example.com")
        assert client.get("/api/plans", headers=headers).json()["simple"]["count"] == 0


class TestRouting:
    @pytest.mark.parametrize("with_profile", [False, True])
    def test_optional_null_housing_inputs_match_omission(self, client, with_profile):
        from finrec.profile import Profile
        payload = {"home_price": 500000, "location": "Austin, TX", "live": False}
        if with_profile:
            payload["profile"] = Profile().to_dict()
        omitted = client.post("/api/buy-vs-rent", json=payload)
        inferred = client.post("/api/buy-vs-rent",
                               json={**payload, "mortgage_rate": None, "monthly_rent": None})
        assert omitted.status_code == inferred.status_code == 200
        assert omitted.json()["simple"] == inferred.json()["simple"]
        assert omitted.json()["advanced"]["inputs"] == inferred.json()["advanced"]["inputs"]

    def test_plan_routes_are_not_swallowed_by_the_catch_all(self, client, sample):
        """`/api/{analysis:path}` matches anything; order is what saves us."""
        client.post("/api/plans", json=sample)
        assert "version" in client.post("/api/plans", json=sample).json()["simple"]

    def test_analysis_routes_still_work(self, client):
        body = client.post("/api/dashboard", json={"salary": 150_000, "location": "Austin, TX"})
        assert body.status_code == 200
        assert "simple" in body.json()

    def test_unknown_analysis_is_404(self, client):
        assert client.post("/api/not-a-thing", json={}).status_code == 404


class TestEnvelope:
    @pytest.mark.parametrize("path", ["/api/plans", "/api/export"])
    def test_responses_keep_the_standard_shape(self, client, sample, path):
        client.post("/api/plans", json=sample)
        body = client.get(path).json()
        for key in ("simple", "advanced", "assumptions", "meta"):
            assert key in body

    def test_responses_are_json_safe(self, client, sample):
        """No NaN or inf may reach a client — JSON has no way to express them."""
        import json

        client.post("/api/plans", json=sample)
        raw = client.get("/api/plans/my-plan").text
        assert "NaN" not in raw and "Infinity" not in raw
        json.loads(raw)
