"""Mapping rules on synthetic feature lists."""

import pytest

from har_repro.features.mapping import build_mapping

from conftest import SYN_FEATURES

SELECTED = ["aAcc-mean()-Z", "fB-bandsEnergy()-1,8", "aAcc-mean()-X", "mag-sma()"]   # deliberately unsorted


@pytest.mark.parametrize("variant,expected_band_index", [("X", 4), ("Y", 6), ("Z", 8)])
def test_variant_selects_nth_occurrence(variant, expected_band_index):
    m = build_mapping(SELECTED, SYN_FEATURES, variant)
    assert m.ambiguous[0].candidates == (4, 6, 8)
    assert m.ambiguous[0].chosen == expected_band_index
    assert m.n_unique == 3 and len(m.ambiguous) == 1


def test_columns_are_in_ascending_original_order_not_selection_order():
    m = build_mapping(SELECTED, SYN_FEATURES, "X")
    assert m.indices_1based == (1, 3, 4, 11)
    assert m.indices_0based == (0, 2, 3, 10)
    assert m.names == ("aAcc-mean()-X", "aAcc-mean()-Z", "fB-bandsEnergy()-1,8", "mag-sma()")


def test_missing_name_raises():
    with pytest.raises(KeyError):
        build_mapping(["not-a-feature"], SYN_FEATURES)


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_mapping(SELECTED, SYN_FEATURES, "W")


def test_name_with_unexpected_multiplicity_raises():
    with pytest.raises(ValueError):
        build_mapping(["dup"], ["dup", "x", "dup"])   # occurs twice: neither unique nor a triplicate


def test_to_dict_is_complete():
    d = build_mapping(SELECTED, SYN_FEATURES, "Y").to_dict()
    assert d["n_selected"] == 4 and d["column_order"] == "ascending_original_index"
    assert d["ambiguous"] == [{"name": "fB-bandsEnergy()-1,8", "candidates_1based": [4, 6, 8], "chosen_1based": 6}]
