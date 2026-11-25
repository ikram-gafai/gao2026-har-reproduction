"""Structural integrity of the real, checksum-verified UCI HAR data (no model fitting, no predictions)."""

from collections import Counter

import numpy as np
import pytest

from har_repro.config import load_experiment_config
from har_repro.data.loader import encode_labels

pytestmark = pytest.mark.data
CFG = load_experiment_config()["data"]


def test_shapes_and_feature_indices(uci):
    assert uci.X_train.shape == (CFG["n_train"], CFG["n_features"]) == (7352, 561)
    assert uci.X_test.shape == (CFG["n_test"], CFG["n_features"]) == (2947, 561)
    assert uci.features.indices == tuple(range(1, 562))


def test_feature_name_structure(uci):
    counts = Counter(uci.features.names)
    dup = {n: c for n, c in counts.items() if c > 1}
    assert len(counts) == 477
    assert len(dup) == 42 and set(dup.values()) == {3}
    assert all("bandsEnergy" in n for n in dup)


def test_participants_disjoint_and_counted(uci):
    tr, te = set(uci.subject_train.tolist()), set(uci.subject_test.tolist())
    assert len(tr) == CFG["n_train_subjects"] == 21
    assert len(te) == CFG["n_test_subjects"] == 9
    assert tr.isdisjoint(te)
    assert tr | te == set(range(1, 31))
    assert te == {2, 4, 9, 10, 12, 13, 18, 20, 24}


def test_labels_and_encoding(uci):
    assert [uci.activity_names[i] for i in range(1, 7)] == CFG["activity_names"]
    assert dict(Counter(uci.y_train.tolist())) == {1: 1226, 2: 1073, 3: 986, 4: 1286, 5: 1374, 6: 1407}
    # Test-class totals equal the denominators implied by the paper's compact confusion matrix (Section 3.3).
    assert dict(Counter(uci.y_test.tolist())) == {1: 496, 2: 471, 3: 420, 4: 491, 5: 532, 6: 537}
    enc = encode_labels(uci.y_test, uci.activity_names)
    assert sorted(set(enc.tolist())) == [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(enc + 1, uci.y_test)


def test_values_finite_and_pre_normalised(uci):
    for X in (uci.X_train, uci.X_test):
        assert np.isfinite(X).all()
        assert X.min() == -1.0 and X.max() == 1.0


def test_no_duplicate_rows_within_or_across_partitions(uci):
    tr = {r.tobytes() for r in uci.X_train}
    te = {r.tobytes() for r in uci.X_test}
    assert len(tr) == len(uci.X_train) and len(te) == len(uci.X_test)
    assert tr.isdisjoint(te)


def test_parsed_values_bitwise_equal_exact_decimal_parsing(uci, verified_root):
    for part, X in (("train", uci.X_train), ("test", uci.X_test)):
        text = (verified_root / part / f"X_{part}.txt").read_text(encoding="ascii")
        exact = np.array([[float(t) for t in line.split()] for line in text.splitlines() if line.strip()])
        assert np.array_equal(X, exact)


def test_partition_order_is_unchanged(uci):
    # The official files are ordered by participant blocks; the loader must not shuffle or sort rows.
    assert uci.subject_test[0] == 2 and uci.subject_test[-1] == 24
    assert uci.subject_train[0] == 1 and uci.subject_train[-1] == 30
