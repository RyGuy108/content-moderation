"""Turn detector scores into allow / review / block, per tenant.

Detectors only produce scores. Everything a business would argue about (is swimwear OK? do we
let a licensed merch shop render its own logo? does an adult platform allow nudity?) lives in
policy config, so changing it never needs a retrain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ALLOW, REVIEW, BLOCK = "allow", "review", "block"
_RANK = {ALLOW: 0, REVIEW: 1, BLOCK: 2}

# Labels no tenant can turn off.
NEVER_ALLOWED = {"minor_sexual", "csam_risk", "real_person_sexual"}


@dataclass
class Rule:
    review: float | None   # score >= review -> at least REVIEW (None: label switched off for this tenant)
    block: float | None    # score >= block  -> BLOCK (None: never auto-blocks, review only)


@dataclass
class Policy:
    name: str = "default"
    version: str = "1"
    rules: dict[str, Rule] = field(default_factory=dict)
    obfuscation_escalates: bool = True
    obfuscation_floor: float = 0.05     # an obfuscated prompt with any risky score above this -> review

    @classmethod
    def load(cls, path: str | Path, tenant: str | None = None) -> "Policy":
        cfg = yaml.safe_load(Path(path).read_text())
        base = cfg["default"]
        over = (cfg.get("tenants") or {}).get(tenant or "", {})
        rules = {}
        for label, r in {**base["rules"], **over.get("rules", {})}.items():
            if label in NEVER_ALLOWED and label in over.get("rules", {}):
                r = base["rules"][label]          # tenants cannot loosen these
            rules[label] = Rule(review=r["review"], block=r.get("block"))
        return cls(name=tenant or "default", version=str(cfg.get("version", "1")), rules=rules,
                   obfuscation_escalates=over.get("obfuscation_escalates", base.get("obfuscation_escalates", True)),
                   obfuscation_floor=base.get("obfuscation_floor", 0.05))


@dataclass
class Decision:
    action: str
    reasons: list[dict]
    scores: dict[str, float]

    def to_dict(self) -> dict:
        return {"action": self.action, "reasons": self.reasons, "scores": self.scores}


def combine(scores: dict[str, float]) -> dict[str, float]:
    """Derived labels: a sexual depiction of a real person is non-consensual intimate imagery,
    and sexual content with any minor signal is treated as CSAM risk."""
    s = dict(scores)
    s["real_person_sexual"] = min(s.get("real_person", 0.0), s.get("sexual", 0.0))
    s["csam_risk"] = min(max(s.get("minor_present", 0.0), s.get("minor_sexual", 0.0)), max(s.get("sexual", 0.0), s.get("minor_sexual", 0.0)))
    return s


def decide(policy: Policy, scores: dict[str, float], flags: set[str] | None = None,
           evidence: dict[str, dict] | None = None) -> Decision:
    scores = combine(scores)
    action, reasons = ALLOW, []
    for label, rule in policy.rules.items():
        sc = scores.get(label)
        if sc is None:
            continue
        lvl = ALLOW
        if rule.block is not None and sc >= rule.block:
            lvl = BLOCK
        elif rule.review is not None and sc >= rule.review:
            lvl = REVIEW
        if lvl != ALLOW:
            reasons.append({"label": label, "score": round(sc, 4), "action": lvl,
                            **({"evidence": evidence[label]} if evidence and label in evidence else {})})
            if _RANK[lvl] > _RANK[action]:
                action = lvl
    if flags and policy.obfuscation_escalates and action == ALLOW:
        risky = max((v for k, v in scores.items() if k in policy.rules), default=0.0)
        if risky >= policy.obfuscation_floor:
            action = REVIEW
            reasons.append({"label": "obfuscation", "flags": sorted(flags), "action": REVIEW,
                            "score": round(risky, 4)})
    reasons.sort(key=lambda r: (-_RANK[r["action"]], -r.get("score", 0)))
    return Decision(action, reasons, {k: round(v, 4) for k, v in scores.items()})
