"""Pre-registered outcome categories (protocol v2 section 6 with Addendum A1-R1 precedence).

Order (the first rule that matches decides):
1. not_reproduced if the reference-seed gap (Full - Compact) is <= 0 pp
2. strong   : both reference accuracies within +/-1.0 pp of published, and the gap > 0 and within +/-1.0 pp of +1.66
3. partial  : Full within +/-1.0 pp, or at least one model within +/-2.0 pp with a positive gap
4. not_reproduced otherwise
"Within" is inclusive. A 1e-9 pp slack absorbs floating-point representation error only.
"""

from __future__ import annotations

from typing import Any, Mapping

EPS = 1e-9


def _within(x: float, tol: float) -> bool:
    return abs(x) <= tol + EPS


def classify_outcome(full_acc: float, compact_acc: float, published: Mapping[str, Any],
                     rule: Mapping[str, Any]) -> dict:
    d_full = 100.0 * (full_acc - published["full561_accuracy"])
    d_comp = 100.0 * (compact_acc - published["compact100_accuracy"])
    gap = 100.0 * (full_acc - compact_acc)
    d_gap = gap - published["gap_pp"]
    detail = {"full_minus_published_pp": d_full, "compact_minus_published_pp": d_comp,
              "gap_pp": gap, "gap_minus_published_pp": d_gap}

    if gap <= 0 + EPS:
        return {"category": "not_reproduced", "rule": "reference gap <= 0 pp (A1-R1 precedence)", **detail}
    if (_within(d_full, rule["strong_model_tolerance_pp"]) and _within(d_comp, rule["strong_model_tolerance_pp"])
            and _within(d_gap, rule["strong_gap_tolerance_pp"])):
        return {"category": "strong", "rule": "both models and gap within +/-1.0 pp", **detail}
    matched = []
    if _within(d_full, rule["partial_full_tolerance_pp"]):
        matched.append("full561 within +/-1.0 pp")
    if _within(d_full, rule["partial_any_tolerance_pp"]):
        matched.append("full561 within +/-2.0 pp with positive gap")
    if _within(d_comp, rule["partial_any_tolerance_pp"]):
        matched.append("compact100 within +/-2.0 pp with positive gap")
    if matched:
        return {"category": "partial", "rule": "; ".join(matched), **detail}
    return {"category": "not_reproduced", "rule": "no criterion met", **detail}
