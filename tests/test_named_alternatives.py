import pytest

from finrec.analysis import named_scenarios_service, scenario_housing_analysis
from finrec.profile import Profile
from finrec.scenario import Scenario, ScenarioAssumptions, HomePurchase, OwnedProperty


def test_named_alternatives_share_market_assumptions(monkeypatch):
    profile = Profile()
    profile.saved_scenarios = {
        "Stay": Scenario(name="Stay", assumptions=ScenarioAssumptions(expected_return=.20)).to_dict(),
        "Move": Scenario(name="Move", assumptions=ScenarioAssumptions(expected_return=.01)).to_dict(),
    }
    captured = []

    def compare(p, scenarios, **kwargs):
        captured.extend(scenarios)
        return {"results": [
            {"median_net_worth": [100., 100.], "median_shortfall": [0., 0.],
             "lowest_savings": 5., "fi_age": None} for _ in scenarios]}

    monkeypatch.setattr("finrec.analysis.compare_scenarios", compare)
    result = named_scenarios_service({"profile": profile.to_dict(), "names": ["Stay", "Move"],
                                     "years": 10, "assumptions": {"expected_return": .04}})
    assert len(result["simple"]["comparisons"]) == 3
    assert all(s.assumptions.expected_return == .04 for s in captured)
    assert profile.saved_scenarios["Stay"]["assumptions"]["expected_return"] == .20


def test_housing_helpers_keep_properties_and_overrides(monkeypatch):
    profile = Profile(home_value=500_000)
    scenario = Scenario(
        new_home=HomePurchase(price=600_000),
        properties=[OwnedProperty(value=300_000)],
        assumptions=ScenarioAssumptions(expected_return=.02),
    )
    captured = []

    def buy(p, home, candidate, **kwargs):
        captured.append(candidate)
        return {"breakeven_rent": 1234}

    monkeypatch.setattr("finrec.analysis.breakeven_rent_for_buying", buy)
    result = scenario_housing_analysis(profile, scenario, 20)
    assert result["buy_or_rent"]["breakeven_rent"] == 1234
    assert captured[0].properties == scenario.properties
    assert captured[0].assumptions == scenario.assumptions
    assert scenario.new_home is not None


def test_unknown_alternative_and_unbounded_simulation_rejected():
    p = Profile().to_dict()
    with pytest.raises(ValueError):
        named_scenarios_service({"profile": p, "names": ["missing"]})
    with pytest.raises(ValueError):
        named_scenarios_service({"profile": p, "n_sims": 100_000_000})


@pytest.mark.parametrize("extra", [
    {"years": 5.5}, {"n_sims": True}, {"names": "Stay"},
    {"names": ["Stay", "Stay"]}, {"assumptions": {"expected_return": float("nan")}},
    {"assumptions": {"not_an_assumption": .05}},
])
def test_comparison_rejects_invalid_inputs(extra):
    profile = Profile()
    profile.saved_scenarios = {"Stay": Scenario(name="Stay").to_dict()}
    with pytest.raises(ValueError):
        named_scenarios_service({"profile": profile.to_dict(), "names": ["Stay"], **extra})


def test_sensitivity_samples_shared_inputs_and_labels_its_limit(monkeypatch):
    from finrec.analysis import scenario_sensitivity_service

    captured = []

    def compare(payload):
        captured.append(payload)
        return {"simple": {"comparisons": [
            {"scenario": "Baseline", "terminal_net_worth": 10},
            {"scenario": "Move", "terminal_net_worth": 12},
        ]}, "assumptions": []}

    monkeypatch.setattr("finrec.analysis.named_scenarios_service", compare)
    result = scenario_sensitivity_service({"profile": Profile().to_dict(),
                                           "names": ["Move"], "years": 20})
    assert len(captured) == 5
    assert all(payload["names"] == ["Move"] and payload["years"] == 20 for payload in captured)
    assert all(row["leading_scenario"] == "Move" for row in result["simple"]["samples"])
    assert "not an exact" in result["meta"]["note"]
