from pathlib import Path

from modpipe.policy import Policy, decide

POLICY = Path(__file__).resolve().parents[1] / "policy.yaml"


def test_thresholds():
    p = Policy.load(POLICY)
    assert decide(p, {"sexual": 0.9}).action == "block"
    assert decide(p, {"sexual": 0.5}).action == "review"
    assert decide(p, {"sexual": 0.1}).action == "allow"
    assert decide(p, {"real_person": 0.99}).action == "review"      # likeness alone never auto-blocks


def test_derived_labels():
    p = Policy.load(POLICY)
    d = decide(p, {"real_person": 0.9, "sexual": 0.5})
    assert d.action == "block" and d.reasons[0]["label"] == "real_person_sexual"


def test_tenant_overrides_and_floor():
    adult = Policy.load(POLICY, "adult-platform")
    assert decide(adult, {"sexual": 0.95}).action == "allow"
    # tenants cannot loosen minor_sexual even if their config tries to
    assert decide(adult, {"minor_sexual": 0.9}).action == "block"
    assert decide(adult, {"real_person": 0.9, "sexual": 0.9}).action == "block"
    merch = Policy.load(POLICY, "licensed-merch")
    assert decide(merch, {"trademark": 0.9, "logo": 0.9}).action == "allow"


def test_obfuscation_escalates():
    p = Policy.load(POLICY)
    assert decide(p, {"sexual": 0.1}, {"leet"}).action == "review"
    assert decide(p, {"sexual": 0.01}, {"leet"}).action == "allow"
