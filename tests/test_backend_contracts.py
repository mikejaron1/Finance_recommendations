"""Regression contracts for strict boundaries, identity, backup and consent."""

import json
import sqlite3
import time
from dataclasses import replace

import pytest

from finrec import advisor_llm, db, ingest, notes_review, providers, service, storage
from finrec.profile import Profile
from finrec.validation import profile_from_payload, validate_profile


@pytest.mark.parametrize("field,value", [
    ("cash", "bad"), ("cash", float("inf")), ("cash", float("nan")),
    ("cash", -1), ("age", 35.5), ("age", True), ("filing_status", "bogus"),
    ("retirement_age", 20), ("life_expectancy", 40), ("mortgage_rate", 6.5),
    ("context_notes", []), ("action_states", []), ("state", "XX"),
    ("college_plans", [{"balance": "broken"}]),
    ("properties", [{"mortgage_rate": 9}]),
    ("active_scenario", {"one_offs": [{"amount": "bad"}]}),
    ("active_scenario", {"new_home": {"term_years": 1000000}}),
    ("active_scenario", {"assumptions": {"safe_withdrawal_rate": 0}}),
])
def test_invalid_profile_rejected_before_save(field, value):
    with pytest.raises(ValueError):
        profile_from_payload({field: value})
    profile = Profile()
    setattr(profile, field, value)
    with pytest.raises(ValueError):
        storage.save_plan(profile)
    assert storage.list_plans() == []


def test_profile_metadata_survives_all_boundaries():
    p = Profile(action_states={"a": {"status": "completed"}},
                saved_scenarios={"move": {"name": "Move"}},
                input_provenance={"cash": {"kind": "outdated", "source": "statement", "as_of": "2025"}})
    plan = storage.save_plan(p)
    assert storage.load_plan(plan_id=plan.id).to_dict() == p.to_dict()
    assert profile_from_payload(p.to_dict()).to_dict() == p.to_dict()


def test_zero_mortgage_years_is_valid_only_without_outstanding_debt():
    profile = profile_from_payload({
        "mortgage_balance": 0, "mortgage_years_remaining": 0,
        "properties": [{"value": 100000, "mortgage_balance": 0, "mortgage_years_remaining": 0}],
        "active_scenario": {"properties": [
            {"value": 100000, "mortgage_balance": 0, "mortgage_years_remaining": 0}]},
    })
    saved = storage.save_plan(profile)
    assert storage.load_plan(plan_id=saved.id).mortgage_years_remaining == 0
    for payload in (
        {"mortgage_balance": 100000, "mortgage_years_remaining": 0},
        {"properties": [{"mortgage_balance": 100000, "mortgage_years_remaining": 0}]},
        {"active_scenario": {"properties": [{"mortgage_balance": 100000, "mortgage_years_remaining": 0}]}},
    ):
        with pytest.raises(ValueError, match="outstanding"):
            profile_from_payload(payload)


def test_scenario_event_years_remain_valid_outside_chart_without_clamping():
    scenario = {
        "current_home": {"action": "rent_out", "year": 150},
        "spending_changes": [{"monthly_amount": -100, "start_year": 200, "end_year": 250}],
        "income_changes": [{"new_gross_income": 1000, "start_year": 300}],
        "one_offs": [{"amount": 1000, "year": 500}],
        "assumptions": {"inflation": -0.99, "safe_withdrawal_rate": 0.00000001},
    }
    profile = profile_from_payload({"active_scenario": scenario})
    assert profile.active_scenario == scenario


@pytest.mark.parametrize("scenario", [
    {"properties": [{"action": "rent_out"}]},
    {"current_home": {"action": "invalid"}},
    {"new_home": {"year": True}},
    {"new_home": {"year": "1"}},
    {"new_home": {"year": 1.0}},
    {"one_offs": [{"loan_years": 0}]},
    {"one_offs": [{"amount": -1}]},
    {"spending_changes": [{"start_year": 20, "end_year": 19}]},
    {"assumptions": {"inflation": -1}},
    {"assumptions": {"borrow_rate": -0.01}},
])
def test_nested_scenario_runtime_validation(scenario):
    with pytest.raises(ValueError):
        profile_from_payload({"active_scenario": scenario})


@pytest.mark.parametrize("metadata", [
    {"action_states": {"a": []}},
    {"action_states": {"a": {"status": []}}},
    {"action_states": {"a": {"status": "done"}}},
    {"action_states": {"a": {"status": "completed", "reason": 2}}},
    {"action_states": {"a": {"status": "completed", "updated_at": "yesterday"}}},
    {"action_states": {"a": {"status": "snoozed", "until": "tomorrow"}}},
    {"action_states": {"a": {"status": "not_applicable"}}},
    {"action_states": {"a": {"status": "planned", "unexpected": "x"}}},
    {"input_provenance": {"cash": {"kind": []}}},
    {"input_provenance": {"cash": {"kind": "made_up"}}},
    {"input_provenance": {"cash": {"source": 123}}},
    {"input_provenance": {"cash": {"as_of": 2026}}},
])
def test_action_and_provenance_metadata_is_validated(metadata):
    with pytest.raises(ValueError):
        profile_from_payload(metadata)


def test_new_services_are_registered_and_require_explicit_inputs():
    assert {"actions", "scenarios/compare", "scenarios/sensitivity"} <= service.SERVICES.keys()
    for name in ("actions", "scenarios/compare", "scenarios/sensitivity"):
        with pytest.raises(ValueError, match="profile"):
            service.SERVICES[name]({})
    profile = Profile(saved_scenarios={"Alternative": {}})
    for name in ("scenarios/compare", "scenarios/sensitivity"):
        with pytest.raises(ValueError, match="names"):
            service.SERVICES[name]({"profile": profile.to_dict()})
        with pytest.raises(ValueError, match="names"):
            service.SERVICES[name]({"profile": profile.to_dict(), "names": []})


def test_identity_is_not_display_name_or_slug_normalization():
    a = storage.save_plan(Profile(cash=10), "Plan A", create=True)
    b = storage.save_plan(Profile(cash=20), "Plan-A", create=True)
    assert a.id != b.id and a.slug != b.slug
    renamed = storage.save_plan(Profile(cash=30), "Renamed", plan_id=a.id, expected_version=1)
    assert renamed.id == a.id and renamed.slug == a.slug and renamed.version == 2
    assert storage.load_plan(a.slug).cash == 30
    assert storage.load_plan(plan_id=b.id).cash == 20
    c = storage.save_plan(Profile(), "Renamed", create=True)
    assert c.id != a.id
    with pytest.raises(storage.ConflictError, match="Multiple"):
        storage.save_plan(Profile(), "Renamed")


def test_stale_writes_and_deletes_never_destroy_newer_work():
    saved = storage.save_plan(Profile(), "A", create=True)
    storage.save_plan(Profile(cash=99), "A", plan_id=saved.id, expected_version=1)
    with pytest.raises(storage.ConflictError):
        storage.save_plan(Profile(cash=1), "A", plan_id=saved.id, expected_version=1)
    with pytest.raises(storage.ConflictError):
        storage.delete_plan(plan_id=saved.id, expected_version=1)
    assert storage.load_plan(plan_id=saved.id).cash == 99
    assert len(storage.list_versions(saved.slug)) == 2


def test_dirty_page_input_merges_preserve_other_tabs_and_financial_versions():
    from concurrent.futures import ThreadPoolExecutor

    plan = storage.save_plan(Profile(), "A")
    storage.merge_page_inputs(plan.id, {"existing": {"v": 10, "seed": 1, "t": 1, "n": 1}})
    def update(index):
        storage.merge_page_inputs(plan.id, {f"page.{index}": {"v": index, "seed": 0, "t": 2, "n": index}})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(update, range(8)))
    result = storage.load_page_inputs(plan.id)
    assert result["existing"]["v"] == 10
    assert all(result[f"page.{i}"]["v"] == i for i in range(8))
    assert result["__version__"] == 2
    assert len(storage.list_versions(plan.slug)) == 1


def test_page_input_cap_and_ownership():
    owner, other = db.ensure_user("owner@fixture"), db.ensure_user("other@fixture")
    plan = storage.save_plan(Profile(), user_id=owner)
    updates = {f"p.{i}": {"v": i, "t": i, "n": i} for i in range(2005)}
    merged = storage.merge_page_inputs(plan.id, updates, user_id=owner)
    assert len(merged) == 2001
    assert "p.0" not in merged and "p.2004" in merged
    with pytest.raises(ValueError, match="another user"):
        storage.merge_page_inputs(plan.id, {"bad": 1}, user_id=other)
    with pytest.raises(ValueError, match="2000"):
        storage.merge_page_inputs(plan.id, {}, max_entries=2001, user_id=owner)


def test_legacy_page_inputs_belong_only_to_last_used_plan():
    a = storage.save_plan(Profile(), "A")
    b = storage.save_plan(Profile(), "B")
    user = db.default_user_id()
    db.set_state(user, "page_inputs", json.dumps({"__version__": 2, "old": {"v": 42}}))
    assert storage.load_page_inputs(a.id) == {}
    assert storage.load_page_inputs(b.id)["old"]["v"] == 42
    assert db.get_state(user, "page_inputs") is None
    assert db.get_state(user, "page_inputs_legacy_owner") == str(b.id)


def test_new_plan_save_freezes_legacy_page_input_owner():
    a = storage.save_plan(Profile(), "A")
    user = db.default_user_id()
    db.set_state(user, "page_inputs", json.dumps({"__version__": 2, "old": {"v": 42}}))
    b = storage.save_plan(Profile(), "B", create=True)
    assert storage.load_page_inputs(a.id)["old"]["v"] == 42
    assert storage.load_page_inputs(b.id) == {}


def test_orphaned_legacy_page_inputs_never_attach_to_a_new_plan():
    user = db.default_user_id()
    db.set_state(user, "page_inputs", json.dumps({"__version__": 2, "old": {"v": 42}}))
    plan = storage.save_plan(Profile(), "A")
    assert storage.load_page_inputs(plan.id) == {}
    assert db.get_state(user, "page_inputs_legacy_owner") == '"unassigned"'


def test_corrupt_optional_legacy_cache_never_blocks_financial_saves():
    profile = Profile()
    plan = storage.save_plan(profile, "A")
    user = db.default_user_id()
    db.set_state(user, "page_inputs", "not-json")
    saved = storage.save_plan(replace(profile, cash=12345), "A",
                              plan_id=plan.id, expected_version=1)
    assert saved.version == 2
    assert storage.load_plan(plan_id=plan.id).cash == 12345
    assert db.get_state(user, "page_inputs") == "not-json"
    warning = storage.page_inputs_migration_status()
    assert warning["status"] == "corrupt" and warning["owner_plan_id"] == plan.id
    with pytest.raises(ValueError, match="financial profiles remain saved"):
        storage.load_page_inputs(plan.id)
    storage.merge_page_inputs(plan.id, {"new.field": {"v": 1}})
    assert storage.load_page_inputs(plan.id)["new.field"]["v"] == 1
    assert storage.export_all()["user_state"]["page_inputs"] == "not-json"
    storage.delete_plan(plan_id=plan.id, purge=True)
    assert db.get_state(user, "page_inputs") is None
    assert storage.page_inputs_migration_status() == {}


def test_corrupt_orphan_cache_is_preserved_without_attaching_to_first_plan():
    user = db.default_user_id()
    db.set_state(user, "page_inputs", "not-json")
    storage.migrate_legacy_inputs()
    plan = storage.save_plan(Profile(), "A")
    assert storage.load_page_inputs(plan.id) == {}
    assert db.get_state(user, "page_inputs") == "not-json"
    assert storage.page_inputs_migration_status()["owner_plan_id"] is None


def test_page_inputs_follow_remapped_backup_plan_ids():
    plan = storage.save_plan(Profile(), "A")
    storage.merge_page_inputs(plan.id, {"old": {"v": 42}})
    restored = storage.import_all(storage.export_all())
    copied = restored["id_mapping"][0]["id"]
    assert copied != plan.id
    assert storage.load_page_inputs(copied)["old"]["v"] == 42


def test_permanent_plan_deletion_removes_scoped_page_and_tailoring_state():
    profile = Profile(context_notes="A note")
    plan = storage.save_plan(profile, "A")
    storage.merge_page_inputs(plan.id, {"balance": {"v": 123}})
    storage.save_tailoring([], notes=profile.context_notes, provider="fixture",
                           profile=profile, plan_id=plan.id)
    storage.delete_plan(plan_id=plan.id, purge=True)
    state = storage.export_all()["user_state"]
    assert f"page_inputs:{plan.id}" not in state
    assert f"tailored_advice:{plan.id}" not in state


def test_other_users_cannot_address_numeric_plan_ids():
    a, b = db.ensure_user("a@fixture"), db.ensure_user("b@fixture")
    saved = storage.save_plan(Profile(), user_id=a)
    assert storage.load_plan(plan_id=saved.id, user_id=b) is None
    with pytest.raises(storage.ConflictError):
        storage.save_plan(Profile(), plan_id=saved.id, user_id=b)


def test_full_backup_restores_history_beyond_50_and_deleted_plans():
    saved = storage.save_plan(Profile(cash=1), "History", create=True)
    for number in range(2, 62):
        storage.save_plan(Profile(cash=number), "History", plan_id=saved.id)
    deleted = storage.save_plan(Profile(), "Deleted", create=True)
    storage.delete_plan(plan_id=deleted.id)
    backup = storage.export_all()
    assert len(backup["plans"][0]["history"]) == 61
    assert backup["plans"][1]["deleted_at"] is not None
    preview = storage.import_all(backup, dry_run=True)
    assert preview["versions"] == 62
    assert len(storage.list_plans()) == 1
    with pytest.raises(storage.ConflictError):
        storage.import_all(backup, collision="error")
    restored = storage.import_all(backup)
    assert len(restored["id_mapping"]) == 2
    assert storage.load_version(restored["id_mapping"][0]["slug"], 1).cash == 1
    assert storage.load_plan(restored["id_mapping"][0]["slug"]).cash == 61
    assert storage.load_plan(restored["id_mapping"][1]["slug"]) is None
    assert len(storage.export_all()["plans"]) == 4
    assert len(storage.list_versions(saved.slug, limit=500)) == 61


def test_import_validation_is_atomic_and_legacy_limit_is_explicit():
    saved = storage.save_plan(Profile(), "Original")
    backup = storage.export_all()
    backup["plans"].append({"name": "bad", "slug": "bad", "history": [{"version": -1}]})
    with pytest.raises(ValueError):
        storage.import_all(backup)
    assert len(storage.list_plans()) == 1
    legacy = {"format_version": 2, "plans": [
        {"name": "Legacy", "slug": "legacy", "profile": Profile().to_dict(), "history": []}]}
    result = storage.import_all(legacy)
    assert result["legacy_history_unavailable"] is True
    assert storage.load_plan(plan_id=saved.id) is not None


def test_semantic_corruption_recovers_per_version_and_retains_raw_backup():
    saved = storage.save_plan(Profile(cash=11), "A")
    storage.save_plan(Profile(cash=22), "A", plan_id=saved.id)
    broken = json.dumps({"cash": "bad"})
    db.connect().execute("UPDATE plan_versions SET payload = ? WHERE plan_id = ? AND version = 2",
                         (broken, saved.id))
    assert storage.load_plan("a").cash == 11
    with pytest.raises(storage.CorruptPlanError):
        storage.load_version("a", 2)
    backup = storage.export_all()
    assert backup["plans"][0]["history"][1]["payload"] == broken
    restored = storage.import_all(backup)
    assert restored["corrupt_versions"] == 1
    assert storage.load_plan(restored["id_mapping"][0]["slug"]).cash == 11
    db.connect().execute("UPDATE plan_versions SET payload = ? WHERE plan_id = ?", (broken, saved.id))
    with pytest.raises(storage.CorruptPlanError):
        storage.load_plan("a")


def test_legacy_storage_failure_is_retryable(monkeypatch):
    storage.plans_dir().joinpath("legacy.json").write_text(json.dumps(
        {"name": "Legacy", "profile": Profile().to_dict()}))
    original = storage.save_plan
    monkeypatch.setattr(storage, "save_plan", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError()))
    storage._import_legacy_once()
    assert str(db.db_path()) not in storage._legacy_done
    monkeypatch.setattr(storage, "save_plan", original)
    storage._import_legacy_once()
    assert storage.load_plan("legacy") is not None


@pytest.mark.parametrize("payload", [
    {"years": -1}, {"years": 101}, {"years": 1.5}, {"n_sims": 10001},
    {"n_sims": float("inf")}, {"profile": {"cash": "bad"}},
])
def test_simulation_requests_are_bounded(payload):
    with pytest.raises(ValueError):
        service.dashboard_service(payload)


def test_service_uses_canonical_household_tax():
    p = Profile(salary=170000, partner_salary=80000, employment_type="both",
                business_income=30000, annual_401k_contribution=20000)
    expected = p.tax_picture()
    assert service.profile_summary(p)["simple"]["take_home_pay"] == expected.after_tax_income
    assert service.tax_service({"profile": p})["simple"]["total_tax"] == expected.total_tax
    overridden = service.tax_service({"profile": p, "pretax_deferral": 1000})
    assert overridden["simple"]["total_tax"] == p.tax_picture(pretax_deferral=1000).total_tax
    assumption = next(a for a in overridden["assumptions"] if a["name"] == "Pre-tax deferral")
    assert assumption["overridden"] and assumption["unit"] == "currency"
    assert assumption["measurement_unit"] == "USD/year"


def test_assumptions_report_effective_overrides():
    result = service.retirement_service({
        "profile": Profile().to_dict(),
        "overrides": {"annual_contribution": 1000, "expected_return": 0.04},
    })
    assert result["simple"]["annual_contribution"] == 1000
    assert result["meta"]["effective_inputs"]["expected_return"] == 0.04
    assert result["meta"]["effective_inputs"]["spending_in_current_dollars"] is True
    contribution = next(a for a in result["assumptions"] if a["name"] == "Annual contribution")
    assert contribution["value"] == 1000 and contribution["overridden"]
    housing = service.buy_vs_rent_service({"home_price": 500000, "monthly_rent": 1800, "live": False})
    rent = next(a for a in housing["assumptions"] if a["name"] == "Comparable rent")
    assert rent["value"] == 1800 and rent["overridden"] and rent["unit"] == "currency"
    assert rent["measurement_unit"] == "USD/month"


def test_service_assumptions_use_explicit_ui_unit_contract():
    location = service.location_preview("Austin, TX", live=False)
    units = {item["name"]: item["unit"] for item in location["assumptions"]}
    assert units["Price-to-rent ratio"] == "ratio"
    assert units["Property tax rate"] == "percent"
    assert units["Home insurance"] == "currency"
    housing = service.buy_vs_rent_service({"home_price": 500000, "live": False})
    selling = next(item for item in housing["assumptions"] if item["name"] == "Selling costs")
    assert selling["unit"] == "percent"
    tax = service.tax_service({"profile": Profile().to_dict()})
    year = next(item for item in tax["assumptions"] if item["name"] == "Tax year")
    assert year["unit"] == "text" and year["measurement_unit"] == "calendar_year"


def test_tax_scenario_overrides_preserve_earners_and_actual_deductions():
    p = Profile(salary=170000, partner_salary=80000, employment_type="both",
                business_income=30000, partner_business_income=-5000,
                annual_401k_contribution=20000, annual_hsa_contribution=4000,
                above_the_line_deductions=1500, has_hdhp=True)
    original = p.to_dict()
    overrides = {"household_income": p.household_income * 2, "filing_status": "married_joint",
                 "tax_year": 2026, "state": "TX", "itemized": 45000,
                 "long_term_gains": 12000}
    response = service.tax_service({"profile": p.to_dict(), "overrides": overrides})
    result = response["advanced"]
    assert response["meta"]["income_scale"] == 2
    assert result["extra"]["earner_wages"] == [340000, 160000]
    assert result["extra"]["earner_self_employment"] == [60000, -10000]
    assert result["pretax_deferral"] == 24000
    assert result["extra"]["above_the_line"] == 1500
    assert result["deduction_taken"] == 45000
    assert result["state_tax"] == 0
    assert result["capital_gains_tax"] > 0
    assert p.to_dict() == original
    explicit = service.tax_service({"profile": p.to_dict(),
                                    "overrides": {**overrides, "pretax_deferral": 1234}})
    assert explicit["advanced"]["pretax_deferral"] == 1234


def test_tax_partial_overrides_keep_canonical_itemization():
    p = Profile(salary=200000, annual_hsa_contribution=4000, has_hdhp=True,
                annual_401k_contribution=18000, extra_itemized_deductions=50000)
    response = service.tax_service({"profile": p.to_dict(), "overrides": {"long_term_gains": 1000}})
    expected = p.tax_picture(long_term_gains=1000)
    assert response["advanced"]["total_tax"] == expected.total_tax
    assert response["advanced"]["pretax_deferral"] == 22000
    assert response["advanced"]["deduction_type"] == "itemized"


@pytest.mark.parametrize("income,expected_se", [(None, 60000), (120000, 120000)])
def test_tax_itemized_override_includes_self_employed_salary_in_se_aggregate(income, expected_se):
    profile = Profile(salary=50000, business_income=10000, employment_type="self_employed")
    overrides = {"itemized": 0}
    if income is not None:
        overrides["household_income"] = income
    result = service.tax_service({"profile": profile.to_dict(), "overrides": overrides})["advanced"]
    assert result["extra"]["earner_wages"] == [0, 0]
    assert result["extra"]["earner_self_employment"] == [expected_se, 0]
    assert result["extra"]["self_employment_tax"] > 0


@pytest.mark.parametrize("overrides", [
    {"household_income": "bad"}, {"household_income": float("inf")},
    {"pretax_deferral": -1}, {"long_term_gains": -1},
    {"itemized": "bad"}, {"state": "XX"}, {"filing_status": "bogus"},
    {"tax_year": 2050}, {"unknown": 1},
])
def test_invalid_tax_scenario_overrides_are_rejected(overrides):
    with pytest.raises(ValueError):
        service.tax_service({"profile": Profile().to_dict(), "overrides": overrides})


def test_tax_income_scaling_cannot_invent_an_earner_split():
    p = Profile(gross_income=0)
    with pytest.raises(ValueError, match="per-earner"):
        service.tax_service({"profile": p.to_dict(), "overrides": {"household_income": 100000}})


def test_live_false_makes_no_provider_call(monkeypatch):
    monkeypatch.setattr(providers, "current_mortgage_rate",
                        lambda *a, **k: pytest.fail("offline request attempted live fetch"))
    service.location_preview("Austin, TX", live=False)
    service.affordability_service({"profile": Profile(), "live": False})


@pytest.mark.parametrize("with_profile", [False, True])
def test_optional_null_analysis_rates_and_rent_use_inference(with_profile):
    payload = {"home_price": 500000, "location": "Austin, TX",
               "mortgage_rate": None, "monthly_rent": None, "live": False}
    if with_profile:
        payload["profile"] = Profile().to_dict()
    result = service.buy_vs_rent_service(payload)
    assert result["advanced"]["inputs"]["mortgage_rate"] > 0
    assert result["advanced"]["inputs"]["monthly_rent"] > 0
    assert service.affordability_service(payload)["simple"]["safe_max_price"] > 0
    with pytest.raises(ValueError, match="mortgage_rate"):
        profile_from_payload({"mortgage_rate": None})


def test_optional_monthly_spending_null_is_inferred_during_onboarding():
    result = service.create_profile({"salary": 150000, "location": "Austin, TX", "monthly_spending": None})
    assert result["advanced"]["profile"]["monthly_spending"] > 0


def test_cache_validation_failure_backoff_and_observation(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "CACHE_DIR", tmp_path / "cache")
    providers.clear_cache()
    providers._write_cache("mortgage_MORTGAGE30US",
                           {"value": "bad", "fetched_at": time.time()})
    calls = []
    monkeypatch.setattr(providers, "_fetch_fred_series", lambda series: calls.append(series))
    assert providers.current_mortgage_rate() is None
    assert providers.current_mortgage_rate() is None
    assert len(calls) == 1
    providers.clear_cache()
    providers._observed_at["MORTGAGE30US"] = "2026-09-03"
    monkeypatch.setattr(providers, "_fetch_fred_series", lambda series: 6.2)
    assert providers.current_mortgage_rate() == 0.062
    cached = providers._read_cache("mortgage_MORTGAGE30US", 100)
    assert cached["observation_date"] == "2026-09-03"


def test_import_preserves_accounts_and_requires_aggregation():
    found = ingest.extract_from_text("Checking ending 1234 $1000\nSavings ending 5678 $2000")
    assert {f.account for f in found} == {"1234", "5678"}
    with pytest.raises(ValueError, match="aggregation"):
        ingest.apply_findings(Profile(), found, {"cash"})
    result = ingest.apply_findings(Profile(), found, {"cash"}, aggregation={"cash": "sum"})
    assert result.cash == 3000
    assert len(result.input_provenance["cash"]["sources"]) == 2
    pay = ingest.extract_from_text("Base pay biweekly $3000")
    with pytest.raises(ValueError, match="annualize"):
        ingest.apply_findings(Profile(), pay, {"salary"})
    assert ingest.apply_findings(Profile(), pay, {"salary"}, aggregation={"salary": "annualize"}).salary == 78000
    negative = ingest.extract_from_text("Checking -$1000")
    assert negative[0].value == -1000
    with pytest.raises(ValueError):
        ingest.apply_findings(Profile(), negative, {"cash"})


def test_hosted_calls_off_by_default_and_preview_matches_wire(monkeypatch):
    from finrec.recommend import generate_recommendations
    p = Profile(cash=178_321)
    notes = "Holding $178,321 to fund a business."
    recs = generate_recommendations(p)
    monkeypatch.setattr(advisor_llm, "_provider", lambda: ("Fixture", "test", "https://fixture.invalid", "fixture"))
    captured = []
    def post(system, user, *args):
        captured.append(json.loads(user))
        key = "conflicts" if "review" in system.lower() else "insights"
        return {"choices": [{"message": {"content": json.dumps({key: []})}}]}
    monkeypatch.setattr(advisor_llm, "_post_chat", post)
    with pytest.raises(ValueError, match="consent"):
        advisor_llm.llm_insights(p, notes)
    with pytest.raises(ValueError, match="consent"):
        notes_review.llm_conflicts(recs, notes, profile=p)
    assert not captured
    advisor_llm.llm_insights(p, notes, consent=True)
    notes_review.llm_conflicts(recs, notes, profile=p, consent=True)
    assert captured == [advisor_llm.hosted_payload(p, notes, consent=True),
                        advisor_llm.hosted_payload(p, notes, recs=recs, consent=True)]
    assert "178321" not in json.dumps(captured) and "178,321" not in json.dumps(captured)


def test_agreement_does_not_demote_reserve_advice():
    from finrec.recommend import generate_recommendations
    p = Profile(cash=10)
    recs = generate_recommendations(p)
    conflicts = notes_review.local_conflicts(recs, "I deliberately want to build up my emergency fund")
    assert not any(c.action_id == r.action_id for c in conflicts for r in recs if r.category == "Safety")


def test_tailoring_is_scoped_to_plan_profile_model_and_request():
    p = Profile(context_notes="My notes")
    a = storage.save_plan(p, "A")
    b = storage.save_plan(p, "B")
    storage.save_tailoring([], notes=p.context_notes, provider="fixture", model="one",
                           profile=p, plan_id=a.id, request={"task": "both"})
    assert storage.load_tailoring(plan_id=b.id, profile=p) is None
    assert storage.load_tailoring(plan_id=a.id, profile=p, model="one", request={"task": "both"})
    assert storage.load_tailoring(plan_id=a.id, profile=replace(p, cash=123), model="one", request={"task": "both"}) is None
    assert storage.load_tailoring(plan_id=a.id, profile=p, model="two", request={"task": "both"}) is None


def test_api_gate_and_identity_cover_generic_routes(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from finrec.api import app
    remote = TestClient(app, base_url="https://hosted.example", client=("203.0.113.4", 50000))
    assert remote.get("/api/plans").status_code == 403
    monkeypatch.setenv("FINREC_API_TOKEN", "fixture-token")
    monkeypatch.setenv("FINREC_API_USER", "verified@fixture")
    assert remote.get("/api/plans").status_code == 401
    headers = {"Authorization": "Bearer fixture-token"}
    victim = db.ensure_user("victim@fixture")
    body = {"profile": {"cash": 123}, "name": "Owned", "user_id": victim, "email": "victim@fixture"}
    response = remote.post("/api/plans/save", json=body, headers=headers)
    assert response.status_code == 200
    assert storage.list_plans(user_id=victim) == []
    assert remote.get("/api/plans?user=victim@fixture", headers=headers).json()["simple"]["count"] == 1
    assert remote.post("/api/plans", json={"profile": {"cash": "bad"}}, headers=headers).status_code == 422
    updated = {**body, "plan_id": response.json()["simple"]["id"], "expected_version": 0}
    assert remote.post("/api/plans/save", json=updated, headers=headers).status_code == 409
