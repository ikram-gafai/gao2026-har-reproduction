"""Pre-registered outcome categories, including A1-R1 precedence and inclusive boundaries."""

import pytest

from har_repro.config import load_experiment_config
from har_repro.evaluation.decision import classify_outcome

CFG = load_experiment_config()
PUB, RULE = CFG["published"], CFG["decision_rule"]
F, C = PUB["full561_accuracy"], PUB["compact100_accuracy"]


@pytest.mark.parametrize("full,compact,expected", [
    (F, C, "strong"),                                  # exact published values
    (F + 0.010, C + 0.010, "strong"),                  # both +1.0 pp, gap unchanged (inclusive boundary)
    (F + 0.005, C - 0.005, "strong"),                  # gap 2.66 pp: exactly +1.0 pp from 1.66
    (F, C - 0.015, "partial"),                         # compact -1.5 pp: strong fails, full within 1 pp
    (F - 0.015, C - 0.015, "partial"),                 # both -1.5 pp: within 2 pp with positive gap
    (F - 0.025, C - 0.019, "partial"),                 # compact within 2 pp, full not, gap positive
    (F - 0.025, C - 0.025, "not_reproduced"),          # both beyond 2 pp
    (F, F, "not_reproduced"),                          # gap exactly 0 -> A1-R1 precedence
    (F, F + 0.001, "not_reproduced"),                  # full within 1 pp but gap negative -> precedence wins
])
def test_categories(full, compact, expected):
    assert classify_outcome(full, compact, PUB, RULE)["category"] == expected


def test_reported_deltas():
    out = classify_outcome(0.95, 0.94, PUB, RULE)
    assert out["full_minus_published_pp"] == pytest.approx(-0.55)
    assert out["compact_minus_published_pp"] == pytest.approx(0.11)
    assert out["gap_pp"] == pytest.approx(1.0) and out["gap_minus_published_pp"] == pytest.approx(-0.66)
