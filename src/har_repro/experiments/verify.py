"""Audit saved experiment outputs without trusting them.

Checks (all read-only unless --refit is used, which writes only to a temporary directory):
1. every run listed in manifest.json exists, with all files present;
2. proba.npy and predictions.csv agree (labels = argmax of probabilities; CSV probabilities round-trip exactly);
3. if the verified dataset is available: true labels and subject IDs equal the official test partition;
4. metrics.json equals metrics recomputed from the saved predictions;
5. summary.json, bootstrap.json, decision.json and results_table.md equal byte-for-byte the files regenerated
   from the saved predictions (including the 10,000-replicate participant bootstrap);
6. repeatability.json passed, and the fresh-process probabilities are bit-identical to the main runs;
7. optional --refit RUN_ID: retrain that configuration from the public data with the frozen code and require
   bit-identical probabilities.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..data.loader import UCIHAR, encode_labels
from ..evaluation.metrics import classification_metrics
from ..models.train import predict_labels
from .runner import RunSpec, build_plan, compute_summary, execute_run, summary_files


def _spec_from_id(run_id: str, group: str = "verify") -> RunSpec:
    tag, seed = run_id.split("__seed")
    model, _, variant = tag.partition("-")
    return RunSpec(model, variant or None, int(seed), group)


def verify_results(results_dir: Path, cfg: Mapping[str, Any], data: UCIHAR | None = None,
                   mapping_doc: Mapping[str, Any] | None = None, refit: Sequence[str] = (),
                   log: Callable[[str], None] = print) -> dict:
    problems: list[str] = []
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    run_ids = manifest["plan"]
    include_sensitivity = any("-Y__" in r or "-Z__" in r for r in run_ids)
    expected_plan = [s.run_id for s in build_plan(cfg, include_sensitivity=include_sensitivity)]
    if run_ids != expected_plan:
        problems.append("manifest plan differs from the plan implied by the frozen configuration")

    for rid in run_ids:
        d = results_dir / "runs" / rid
        missing = [f for f in ("predictions.csv", "proba.npy", "metrics.json", "run_info.json") if not (d / f).exists()]
        if missing:
            problems.append(f"{rid}: missing {missing}")
            continue
        proba = np.load(d / "proba.npy")
        with open(d / "predictions.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        y_true = np.array([int(r["y_true"]) for r in rows])
        y_pred = np.array([int(r["y_pred"]) for r in rows])
        subjects = np.array([int(r["subject"]) for r in rows])
        pcols = [k for k in rows[0] if k.startswith("p_")]
        csv_proba = np.array([[float(r[k]) for k in pcols] for r in rows], dtype=np.float64)
        if not np.array_equal(predict_labels(proba), y_pred):
            problems.append(f"{rid}: predicted labels are not the argmax of saved probabilities")
        if not np.array_equal(csv_proba, proba.astype(np.float64)):
            problems.append(f"{rid}: CSV probabilities differ from proba.npy")
        if any((r["correct"] == "1") != (int(r["y_true"]) == int(r["y_pred"])) for r in rows):
            problems.append(f"{rid}: 'correct' column inconsistent")
        if data is not None:
            if not np.array_equal(y_true, encode_labels(data.y_test, data.activity_names)):
                problems.append(f"{rid}: true labels differ from the official test partition")
            if not np.array_equal(subjects, data.subject_test):
                problems.append(f"{rid}: subject IDs differ from the official test partition")
        saved = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        class_names = saved["confusion_matrix"]["labels"]
        recomputed = json.loads(json.dumps(classification_metrics(y_true, proba, y_pred, subjects, class_names)))
        if recomputed != saved:
            problems.append(f"{rid}: metrics.json differs from metrics recomputed from predictions")
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        if info["run_id"] != rid or info["n_test"] != len(rows):
            problems.append(f"{rid}: run_info inconsistent")
    log(f"Checked {len(run_ids)} runs (files, label/probability consistency, metrics recomputation)")

    regenerated = summary_files(compute_summary(results_dir, cfg, include_sensitivity))
    for name, text in regenerated.items():
        path = results_dir / "summary" / name
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            problems.append(f"summary/{name} differs from the version regenerated from saved predictions")
    log("Regenerated summary, bootstrap, decision and results table from saved predictions")

    rep_path = results_dir / "repeatability" / "repeatability.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        if not rep.get("passed"):
            problems.append("repeatability.json reports a failure")
        for c in rep.get("checks", []):
            a = np.load(results_dir / "runs" / c["run_id"] / "proba.npy")
            b = np.load(results_dir / "repeatability" / c["run_id"] / "proba.npy")
            if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
                problems.append(f"repeatability {c['run_id']}: saved fresh-process probabilities not bit-identical")
        log(f"Repeatability record checked ({len(rep.get('checks', []))} runs)")
    else:
        problems.append("repeatability/repeatability.json missing")

    refits = []
    for rid in refit:
        if data is None or mapping_doc is None:
            problems.append("--refit requires the verified dataset")
            break
        with tempfile.TemporaryDirectory() as tmp:
            res = execute_run(data, _spec_from_id(rid), cfg, mapping_doc, Path(tmp), True, save_model=False)
        same = np.array_equal(res["proba"].view(np.uint32), np.load(results_dir / "runs" / rid / "proba.npy").view(np.uint32))
        refits.append({"run_id": rid, "bit_identical": bool(same), "fit_seconds": res["run_info"]["fit_seconds"]})
        if not same:
            problems.append(f"refit {rid}: probabilities differ from the saved run")
        log(f"Refit {rid}: {'bit-identical' if same else 'DIFFERENT'} ({res['run_info']['fit_seconds']} s)")

    return {"passed": not problems, "problems": problems, "runs_checked": len(run_ids),
            "manifest_commit": manifest["git"]["commit"], "refits": refits}
