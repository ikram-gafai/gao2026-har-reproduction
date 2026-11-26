"""Audit every number printed in evidence_note.pdf against committed results, the frozen config and paper facts.

Run from the repository root with the paper environment:  python paper/check_numbers.py
Numbers in figures are drawn by make_figures.py directly from the result files; this checks the text layer.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def fmt(x, d=2):
    return f"{x:.{d}f}"


def supported() -> dict[str, str]:
    s = json.loads((R / "summary/summary.json").read_text(encoding="utf-8"))
    b = json.loads((R / "summary/bootstrap.json").read_text(encoding="utf-8"))
    d = json.loads((R / "summary/decision.json").read_text(encoding="utf-8"))
    pub = tomllib.load(open(ROOT / "configs/experiment.toml", "rb"))["published"]
    m = lambda rid: json.loads((R / "runs" / rid / "metrics.json").read_text(encoding="utf-8"))
    ok: dict[str, str] = {}
    add = lambda v, why: ok.setdefault(v, why)
    ref = next(r for r in s["per_seed"] if r["seed"] == 2026)
    fs = s["five_seed"]
    add(fmt(100 * ref["full561_accuracy"]), "seed-2026 full acc"); add(fmt(100 * ref["compact100_accuracy"]), "seed-2026 compact acc")
    add(fmt(ref["gap_pp"]), "seed-2026 gap")
    for k in ("full561_accuracy", "compact100_accuracy"):
        add(fmt(100 * fs[k]["mean"]), f"5-seed mean {k}"); add(fmt(100 * fs[k]["sd"]), f"5-seed SD {k}")
    add(fmt(fs["gap_pp"]["mean"]), "5-seed mean gap"); add(fmt(fs["gap_pp"]["sd"]), "5-seed SD gap")
    for v in b["ci_pp"]:
        add(fmt(v), "seed-2026 bootstrap CI")
    add(str(b["replicates"]).replace("", ""), "bootstrap replicates"); add(str(b["seed"]), "bootstrap seed")
    add(fmt(100 * pub["full561_accuracy"]), "paper full"); add(fmt(100 * pub["compact100_accuracy"]), "paper compact")
    add(fmt(pub["gap_pp"]), "paper gap")
    for v in pub["bootstrap_ci_pp"]:
        add(fmt(v), "paper CI")
    add(fmt(pub["full561_macro_f1"], 4), "paper full F1"); add(fmt(pub["compact100_macro_f1"], 4), "paper compact F1")
    for mdl in ("full561", "compact100-X"):
        ms = [m(f"{mdl}__seed{x}") for x in range(2026, 2031)]
        add(fmt(ms[0]["macro_f1"], 4), f"{mdl} seed-2026 F1")
        add(fmt(sum(q["macro_f1"] for q in ms) / 5, 4), f"{mdl} 5-seed F1")
    for v, a in s["sensitivity_exploratory"].items():
        add(fmt(100 * a["accuracy"]["mean"]), f"{v} mean"); add(fmt(100 * a["accuracy"]["sd"]), f"{v} SD")
        add(fmt(100 * fs["full561_accuracy"]["mean"] - 100 * a["accuracy"]["mean"]), f"{v} mean gap")
    dev = max(abs(d[k]) for k in ("full_minus_published_pp", "compact_minus_published_pp", "gap_minus_published_pp"))
    add(fmt(dev), "largest reference-seed deviation")
    pf, pc = m("full561__seed2026")["per_participant"], m("compact100-X__seed2026")["per_participant"]
    for p in pf:
        add(fmt(100 * (pf[p]["correct"] - pc[p]["correct"]) / pf[p]["total"]), f"participant {p} gap")
    cm = m("compact100-X__seed2026")["confusion_matrix"]["rows_true_cols_pred"]
    add(str(cm[3][4]), "seed-2026 compact Sitting->Standing"); add(str(cm[4][3]), "seed-2026 compact Standing->Sitting")
    # Model configurations from the frozen config (configs/experiment.toml)
    models = tomllib.load(open(ROOT / "configs/experiment.toml", "rb"))["models"]
    for name, mc in models.items():
        for k in ("num_boost_round", "max_depth", "eta", "colsample_bytree", "subsample", "base_score"):
            v = mc[k]
            add(str(v) if isinstance(v, int) else f"{v:.1f}", f"{name}.{k}")
    gvals = fs["gap_pp"]["values"]
    add(fmt(min(gvals)), "min seed gap"); add(fmt(max(gvals)), "max seed gap")
    # Paper facts (Gao et al. 2026 text, Table 1, S2, Sec. 3.1, 3.3) and dataset facts (UCI HAR README)
    for v, why in [("561", "features"), ("100", "compact features"), ("82.17", "paper reduction"), ("0.9085", "paper RFE CV"),
                   ("0.9084", "paper one-SE threshold"), ("80", "paper Sitting->Standing; also tests"), ("29", "paper Standing->Sitting"),
                   ("30", "participants"), ("6", "activities"), ("2.56", "window s"), ("50", "overlap %"), ("7,352", "train windows"),
                   ("21", "train participants"), ("2,947", "test windows"), ("9", "test participants / names"),
                   ("19,683", "3^9"), ("3", "occurrences"), ("1.0", "tolerance"), ("0.5", "base_score"), ("15", "compact runs"),
                   ("95", "distinct inputs"), ("5", "duplicate pairs / seeds"), ("1.7", "approx gap"), ("3.2.0", "xgboost"),
                   ("1.7.7.1", "R xgboost"), ("10,000", "replicates"),
                   # identifiers, not results:
                   ("256", "SHA-256 (hash name)"), ("8", "band name 1-8"), ("16", "band name 1-16"),
                   ("24", "band name 1-24"), ("39", "3^9 superscript as extracted"), ("10", "participant ID 10"),
                   ("4.0", "CC BY 4.0 licence"), ("0", "threshold in 'Delta > 0'")]:
        add(v, why)
    return ok


ALLOWED_CONTEXT = re.compile(r"(doi:\S+|16\(20\):10046|20(13|26)|\b25\b|\b10\b Oct|[0-9a-f]{8}\b|S2|D[123]|\[\d\]|"
                             r"\(i+v?\)|Table \d|Figure \d|^\d\s)")


def main(pdf: str = "paper/evidence_note.pdf") -> int:
    text = "\n".join(p.extract_text() for p in PdfReader(str(ROOT / pdf)).pages)
    ok = supported()
    flagged = []
    for line in text.splitlines():
        cleaned = ALLOWED_CONTEXT.sub(" ", line)
        for tok in re.findall(r"\d[\d,]*(?:\.\d+)*", cleaned):
            tok = tok.rstrip(",.")
            if tok not in ok and tok.replace(",", "") not in ok and tok not in {"1", "2", "4", "7", "2026", "2027", "2028", "2029", "2030"}:
                flagged.append((tok, line.strip()[:90]))
    print(f"{len(ok)} supported values; {len(flagged)} unexplained numeric tokens")
    for tok, line in flagged:
        print(f"  UNEXPLAINED {tok!r} in: {line}")
    return 0 if not flagged else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
