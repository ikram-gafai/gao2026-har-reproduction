"""Loader, label encoding and identical-column detection on a synthetic UCI-HAR-shaped dataset."""

import numpy as np
import pytest

from har_repro.data.loader import (decode_labels, encode_labels, identical_column_groups, load_uci_har,
                                   read_features)


def test_loader_preserves_order_split_values_and_ids(synthetic_root):
    root, arrays = synthetic_root
    d = load_uci_har(root)
    X_tr, y_tr, s_tr = arrays["train"]
    X_te, y_te, s_te = arrays["test"]
    np.testing.assert_array_equal(d.X_train, X_tr)       # exact decimal round-trip, original row order
    np.testing.assert_array_equal(d.X_test, X_te)
    np.testing.assert_array_equal(d.y_train, y_tr)
    np.testing.assert_array_equal(d.subject_test, s_te)
    assert d.X_train.dtype == np.float64
    assert d.features.indices == tuple(range(1, 12))
    assert d.features.names[3] == d.features.names[5] == d.features.names[7]   # duplicates kept, not renamed


def test_label_encoding_round_trip(synthetic_root):
    d = load_uci_har(synthetic_root[0])
    enc = encode_labels(d.y_train, d.activity_names)
    assert enc.min() == 0 and enc.max() == 5
    np.testing.assert_array_equal(enc, d.y_train - 1)
    assert decode_labels(np.array([0, 5]), d.activity_names) == ["WALKING", "LAYING"]
    with pytest.raises(ValueError):
        encode_labels(np.array([1, 7]), d.activity_names)


def test_column_count_mismatch_is_rejected(synthetic_root):
    root, _ = synthetic_root
    f = root / "test" / "X_test.txt"
    f.write_text("\n".join(" ".join(row.split()[:-1]) for row in f.read_text().splitlines()) + "\n")
    with pytest.raises(ValueError):
        load_uci_har(root)


def test_row_count_mismatch_is_rejected(synthetic_root):
    root, _ = synthetic_root
    f = root / "train" / "y_train.txt"
    f.write_text("\n".join(f.read_text().split()[:-1]) + "\n")
    with pytest.raises(ValueError):
        load_uci_har(root)


def test_noncontiguous_feature_indices_are_rejected(tmp_path):
    p = tmp_path / "features.txt"
    p.write_text("1 a\n3 b\n")
    with pytest.raises(ValueError):
        read_features(p)


def test_identical_column_groups(synthetic_root):
    d = load_uci_har(synthetic_root[0])
    assert identical_column_groups(d.X_train) == [[9, 10]]
