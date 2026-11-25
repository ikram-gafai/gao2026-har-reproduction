"""Numerically identical columns: inventory confirmed on training data (and checked to persist in test)."""

import json

import numpy as np
import pytest

from har_repro.cli import MAPPING_FILE
from har_repro.data.loader import identical_column_groups

pytestmark = pytest.mark.data

EXPECTED_COMPACT_PAIRS = [  # 1-based, within the primary (X) compact set
    (202, 215),  # tBodyAccMag-std()        == tGravityAccMag-std()
    (203, 216),  # tBodyAccMag-mad()        == tGravityAccMag-mad()
    (210, 223),  # tBodyAccMag-arCoeff()1   == tGravityAccMag-arCoeff()1
    (227, 232),  # tBodyAccJerkMag-mean()   == tBodyAccJerkMag-sma()
    (503, 508),  # fBodyAccMag-mean()       == fBodyAccMag-sma()
]


@pytest.fixture(scope="module")
def groups(uci):
    return sorted([j + 1 for j in g] for g in identical_column_groups(uci.X_train))


def test_full_set_identical_groups(groups):
    assert len(groups) == 19
    assert sum(len(g) - 1 for g in groups) == 21          # 561 named columns, 540 distinct vectors
    assert [201, 206, 214, 219] in groups


def test_identical_groups_also_identical_in_test(uci, groups):
    for g in groups:
        for j in g[1:]:
            assert np.array_equal(uci.X_test[:, g[0] - 1], uci.X_test[:, j - 1])


@pytest.mark.parametrize("variant", ["X", "Y", "Z"])
def test_compact_sets_contain_exactly_five_identical_pairs(groups, variant):
    sel = set(json.loads(MAPPING_FILE.read_text(encoding="ascii"))["variants"][variant]["indices_1based"])
    pairs = [tuple(i for i in g if i in sel) for g in groups]
    pairs = sorted(p for p in pairs if len(p) > 1)
    assert pairs == EXPECTED_COMPACT_PAIRS
    assert len(sel) - len(pairs) == 95                    # 100 named columns, 95 distinct vectors (none removed)
