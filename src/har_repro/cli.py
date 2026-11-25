"""Command-line entry points.

    python -m har_repro.cli prepare-data [--local-archive PATH]
    python -m har_repro.cli build-mapping [--check]
    python -m har_repro.cli validate-data
    python -m har_repro.cli run-experiments          (Phase 2: all pre-registered runs)
    python -m har_repro.cli run-one --model M [--variant V] --seed N --out DIR
    python -m har_repro.cli summarize
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .config import (DATA_SOURCES_FILE, EXPERIMENT_FILE, canonical_json, config_digest, file_sha256,
                     load_data_sources, load_experiment_config)
from .data.acquire import prepare_dataset, sha256_file, verify_extracted
from .data.loader import identical_column_groups, load_uci_har, read_features
from .features.mapping import build_mapping, read_s2_feature_names
from .paths import CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR, dataset_root, extracted_dir, raw_dir

MAPPING_FILE = CONFIG_DIR / "compact100_mapping.json"
VARIANTS = ("X", "Y", "Z")


def _s2_path(sources: dict) -> Path:
    path = PROJECT_ROOT / sources["supplement"]["local_copy"]
    if sha256_file(path) != sources["supplement"]["sha256"]:
        raise SystemExit(f"S2 copy {path} does not match its recorded SHA-256")
    return path


def cmd_prepare_data(args: argparse.Namespace) -> int:
    sources = load_data_sources()
    root = prepare_dataset(sources, raw_dir(), extracted_dir(), local_archive=args.local_archive)
    print(f"Dataset ready: {root}")
    return 0


def mapping_document() -> dict:
    sources = load_data_sources()
    root = dataset_root()
    verify_extracted(root, {"features.txt": sources["expected"]["files"]["features.txt"]})
    feats = read_features(root / "features.txt")
    s2 = _s2_path(sources)
    names = read_s2_feature_names(s2)
    exp = load_experiment_config()
    doc = {
        "description": ("Gao et al. (2026) Supplementary Table S2 feature names mapped to original UCI HAR "
                        "features.txt indices. Variant X = pre-registered primary (first occurrence of the nine "
                        "triplicated bandsEnergy names); Y and Z = exploratory sensitivity variants. Not a claim "
                        "about which columns the authors used."),
        "primary_variant": exp["mapping"]["primary_variant"],
        "features_txt_sha256": file_sha256(root / "features.txt"),
        "s2_csv_sha256": file_sha256(s2),
        "s2_order": names,
        "variants": {v: build_mapping(names, feats.names, v).to_dict() for v in VARIANTS},
    }
    return doc


def cmd_build_mapping(args: argparse.Namespace) -> int:
    doc = mapping_document()
    text = json.dumps(doc, indent=1, ensure_ascii=True) + "\n"
    if args.check:
        current = MAPPING_FILE.read_text(encoding="ascii") if MAPPING_FILE.exists() else ""
        if current != text:
            print(f"MISMATCH: {MAPPING_FILE} differs from the mapping regenerated from the data", file=sys.stderr)
            return 1
        print(f"OK: {MAPPING_FILE.name} matches the regenerated mapping")
        return 0
    MAPPING_FILE.write_text(text, encoding="ascii", newline="\n")
    for v in VARIANTS:
        m = doc["variants"][v]
        print(f"variant {v}: {m['n_selected']} columns ({m['n_unique_name_matches']} unique + "
              f"{m['n_ambiguous_names']} ambiguous) -> {[a['chosen_1based'] for a in m['ambiguous']]}")
    print(f"Wrote {MAPPING_FILE}")
    return 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Structural validation report. Reads both partitions but fits nothing and predicts nothing."""
    sources = load_data_sources()
    exp = load_experiment_config()
    root = dataset_root()
    verify_extracted(root, sources["expected"]["files"])
    d = load_uci_har(root)
    names = list(d.features.names)
    mapping = json.loads(MAPPING_FILE.read_text(encoding="ascii"))
    groups = identical_column_groups(d.X_train)
    groups_1b = sorted([[j + 1 for j in g] for g in groups])
    test_identical = all(bool(np.array_equal(d.X_test[:, g[0]], d.X_test[:, j])) for g in groups for j in g[1:])
    prim = set(mapping["variants"][exp["mapping"]["primary_variant"]]["indices_1based"])
    compact_pairs = [[i for i in g if i in prim] for g in groups_1b]
    compact_pairs = [p for p in compact_pairs if len(p) > 1]

    # Block -> axis evidence for the nine ambiguous names (training rows only).
    axis = {}
    for sig in ("fBodyAcc", "fBodyAccJerk", "fBodyGyro"):
        energy = {ax: names.index(f"{sig}-energy()-{ax}") for ax in "XYZ"}
        blocks = [i for i, n in enumerate(names) if n == f"{sig}-bandsEnergy()-1,8"]
        axis[sig] = {}
        for k, col in enumerate(blocks, start=1):
            r = {ax: float(np.corrcoef(d.X_train[:, col], d.X_train[:, energy[ax]])[0, 1]) for ax in "XYZ"}
            axis[sig][f"block{k}_col{col + 1}"] = {"r": {a: round(v, 4) for a, v in r.items()},
                                                   "best_axis": max(r, key=r.get)}

    def counts(y):
        u, c = np.unique(y, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "har_repro_version": __version__,
        "python": platform.python_version(),
        "experiment_config_sha256": file_sha256(EXPERIMENT_FILE),
        "experiment_config_digest": config_digest(exp),
        "data_sources_sha256": file_sha256(DATA_SOURCES_FILE),
        "mapping_file_sha256": file_sha256(MAPPING_FILE),
        "extracted_key_files_verified": True,
        "n_features": len(names),
        "feature_indices_contiguous": list(d.features.indices) == list(range(1, len(names) + 1)),
        "n_unique_feature_names": len(set(names)),
        "X_train_shape": list(d.X_train.shape), "X_test_shape": list(d.X_test.shape),
        "y_train_counts": counts(d.y_train), "y_test_counts": counts(d.y_test),
        "activity_names": {int(k): v for k, v in d.activity_names.items()},
        "train_subjects": sorted(set(d.subject_train.tolist())),
        "test_subjects": sorted(set(d.subject_test.tolist())),
        "subject_overlap": sorted(set(d.subject_train.tolist()) & set(d.subject_test.tolist())),
        "nonfinite_values": int((~np.isfinite(d.X_train)).sum() + (~np.isfinite(d.X_test)).sum()),
        "value_range_train": [float(d.X_train.min()), float(d.X_train.max())],
        "value_range_test": [float(d.X_test.min()), float(d.X_test.max())],
        "identical_column_groups_train_1based": groups_1b,
        "identical_groups_also_identical_in_test": test_identical,
        "compact_primary_identical_pairs_1based": compact_pairs,
        "compact_primary_distinct_columns": len(prim) - sum(len(p) - 1 for p in compact_pairs),
        "bandsEnergy_block_axis_evidence_train": axis,
        "note": "Structural checks only; no model was fitted and no prediction was made.",
    }
    out = RESULTS_DIR / "data_validation" / "data_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(canonical_json({k: report[k] for k in ("X_train_shape", "X_test_shape", "subject_overlap",
                                                  "compact_primary_distinct_columns")}))
    print(f"Wrote {out}")
    return 0


# ------------------------------------------------------------------------------------------ Phase 2 commands
def load_dataset(args: argparse.Namespace):
    """Load the dataset; verified against configured SHA-256 unless --allow-unverified-data (tests only)."""
    from .data.acquire import IntegrityError
    root = Path(args.data_root) if args.data_root else dataset_root()
    try:
        verify_extracted(root, load_data_sources()["expected"]["files"])
        verified = True
    except IntegrityError:
        if not args.allow_unverified_data:
            raise SystemExit(f"Dataset at {root} failed checksum verification; run prepare-data. "
                             "(--allow-unverified-data exists for synthetic tests only.)")
        verified = False
    return load_uci_har(root), verified


def _results_dir(args) -> Path:
    return Path(args.results_dir) if args.results_dir else RESULTS_DIR


def cmd_run_one(args: argparse.Namespace) -> int:
    from .experiments.runner import RunSpec, execute_run
    exp = load_experiment_config()
    data, verified = load_dataset(args)
    variant = args.variant if args.model == "compact100" else None
    if args.model == "compact100" and variant is None:
        raise SystemExit("--variant is required for compact100")
    spec = RunSpec(args.model, variant, args.seed, "adhoc")
    mapping = json.loads(MAPPING_FILE.read_text(encoding="ascii"))
    res = execute_run(data, spec, exp, mapping, Path(args.out), verified, save_model=False)
    print(f"{spec.run_id}: wrote {args.out} (fit {res['run_info']['fit_seconds']} s)")
    return 0


def cmd_run_experiments(args: argparse.Namespace) -> int:
    from .experiments.runner import (build_plan, execute_run, git_state, repeatability_check, summarize,
                                     write_manifest)
    exp = load_experiment_config()
    data, verified = load_dataset(args)
    git = git_state()
    if git["source_changes_uncommitted"] and not args.allow_dirty:
        raise SystemExit("Uncommitted changes in src/configs/tests; commit before running (protocol: freeze "
                         "code and configuration before any test prediction).\n" + git["source_changes_uncommitted"])
    if git["commit"] is None and not args.allow_dirty:
        raise SystemExit("Not a git checkout: the frozen commit cannot be recorded (use --allow-dirty to override).")
    results = _results_dir(args)
    if (results / "runs").exists() and any((results / "runs").iterdir()) and not args.overwrite:
        raise SystemExit(f"{results / 'runs'} already contains runs; refusing to overwrite (use --overwrite).")
    plan = build_plan(exp, include_sensitivity=not args.no_sensitivity)
    write_manifest(results, exp, plan, data, verified,
                   {"experiment.toml": EXPERIMENT_FILE, "data_sources.toml": DATA_SOURCES_FILE,
                    "compact100_mapping.json": MAPPING_FILE}, git)
    mapping = json.loads(MAPPING_FILE.read_text(encoding="ascii"))
    for i, spec in enumerate(plan, 1):
        res = execute_run(data, spec, exp, mapping, results / "runs" / spec.run_id, verified)
        m = res["metrics"]
        print(f"[{i}/{len(plan)}] {spec.run_id}: {m['n_correct']}/{m['n']} correct "
              f"({100 * m['accuracy']:.2f}%), fit {res['run_info']['fit_seconds']} s", flush=True)
    out = summarize(results, exp, include_sensitivity=not args.no_sensitivity)
    print(f"Outcome category: {out['decision']['category']} ({out['decision']['rule']})")
    if not args.no_repeatability:
        extra = []
        if args.data_root:
            extra += ["--data-root", str(args.data_root)]
        if args.allow_unverified_data:
            extra += ["--allow-unverified-data"]
        rep = repeatability_check(results, exp, extra)
        print(f"Repeatability (fresh process, seed {rep['seed']}): {'PASSED' if rep['passed'] else 'FAILED'}")
    print(f"Results in {results}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    from .experiments.runner import summarize
    out = summarize(_results_dir(args), load_experiment_config(), include_sensitivity=not args.no_sensitivity)
    print(f"Outcome category: {out['decision']['category']}")
    return 0


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", type=Path, default=None, help="extracted 'UCI HAR Dataset' directory")
    p.add_argument("--allow-unverified-data", action="store_true",
                   help="accept data failing the configured checksums (synthetic tests only; recorded in outputs)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="har_repro", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare-data", help="download (or take a local archive), verify checksums, extract")
    a.add_argument("--local-archive", type=Path, default=None,
                   help="path to an already-downloaded official outer ZIP or inner 'UCI HAR Dataset.zip'")
    a.set_defaults(func=cmd_prepare_data)
    b = sub.add_parser("build-mapping", help="regenerate configs/compact100_mapping.json from the data")
    b.add_argument("--check", action="store_true", help="verify the committed mapping instead of writing it")
    b.set_defaults(func=cmd_build_mapping)
    c = sub.add_parser("validate-data", help="write results/data_validation/data_validation.json")
    c.set_defaults(func=cmd_validate_data)

    r = sub.add_parser("run-experiments", help="Phase 2: all pre-registered runs, summary, bootstrap, repeatability")
    _add_data_args(r)
    r.add_argument("--results-dir", type=Path, default=None)
    r.add_argument("--no-sensitivity", action="store_true", help="skip the exploratory Y/Z mapping runs")
    r.add_argument("--no-repeatability", action="store_true", help="skip the fresh-process repeatability check")
    r.add_argument("--allow-dirty", action="store_true", help="permit uncommitted source/config changes (not for real runs)")
    r.add_argument("--overwrite", action="store_true", help="permit writing over existing run outputs")
    r.set_defaults(func=cmd_run_experiments)

    o = sub.add_parser("run-one", help="fit and evaluate a single configuration (used by the repeatability check)")
    _add_data_args(o)
    o.add_argument("--model", choices=["full561", "compact100"], required=True)
    o.add_argument("--variant", choices=["X", "Y", "Z"], default=None)
    o.add_argument("--seed", type=int, required=True)
    o.add_argument("--out", type=Path, required=True)
    o.set_defaults(func=cmd_run_one)

    s = sub.add_parser("summarize", help="recompute summary, bootstrap and decision from saved runs")
    s.add_argument("--results-dir", type=Path, default=None)
    s.add_argument("--no-sensitivity", action="store_true")
    s.set_defaults(func=cmd_summarize)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
