"""The 100-feature mapping against the official features.txt and the committed manifest."""

import json

import pytest

from har_repro.cli import MAPPING_FILE, mapping_document
from har_repro.config import load_data_sources
from har_repro.data.acquire import sha256_file
from har_repro.features.mapping import build_mapping, read_s2_feature_names
from har_repro.paths import PROJECT_ROOT

pytestmark = pytest.mark.data

# Pre-registered candidate table (protocol v2 section 4): name -> (X/first, Y/second, Z/third), 1-based.
AMBIGUOUS = {
    "fBodyAcc-bandsEnergy()-1,8": (303, 317, 331),
    "fBodyAcc-bandsEnergy()-1,16": (311, 325, 339),
    "fBodyAcc-bandsEnergy()-1,24": (315, 329, 343),
    "fBodyAccJerk-bandsEnergy()-1,8": (382, 396, 410),
    "fBodyAccJerk-bandsEnergy()-1,16": (390, 404, 418),
    "fBodyAccJerk-bandsEnergy()-1,24": (394, 408, 422),
    "fBodyGyro-bandsEnergy()-1,8": (461, 475, 489),
    "fBodyGyro-bandsEnergy()-1,16": (469, 483, 497),
    "fBodyGyro-bandsEnergy()-1,24": (473, 487, 501),
}


@pytest.fixture(scope="module")
def s2_names():
    sup = load_data_sources()["supplement"]
    path = PROJECT_ROOT / sup["local_copy"]
    assert sha256_file(path) == sup["sha256"]
    assert path.stat().st_size == sup["bytes"]
    return read_s2_feature_names(path)


def test_s2_has_100_distinct_names_without_indices(s2_names):
    assert len(s2_names) == 100 == len(set(s2_names))


def test_91_unique_and_9_ambiguous_match_protocol_table(uci, s2_names):
    m = build_mapping(s2_names, uci.features.names, "X")
    assert m.n_unique == 91 and len(m.ambiguous) == 9
    assert {a.name: a.candidates for a in m.ambiguous} == AMBIGUOUS


@pytest.mark.parametrize("variant,pos", [("X", 0), ("Y", 1), ("Z", 2)])
def test_variants_choose_declared_block_and_keep_91_fixed(uci, s2_names, variant, pos):
    m = build_mapping(s2_names, uci.features.names, variant)
    prim = build_mapping(s2_names, uci.features.names, "X")
    assert len(m.indices_1based) == len(set(m.indices_1based)) == 100
    assert list(m.indices_1based) == sorted(m.indices_1based)                 # ascending original order
    assert {a.chosen for a in m.ambiguous} == {c[pos] for c in AMBIGUOUS.values()}
    fixed = set(prim.indices_1based) - {c[0] for c in AMBIGUOUS.values()}
    assert fixed <= set(m.indices_1based) and len(fixed) == 91
    for idx, name in zip(m.indices_1based, m.names):
        assert uci.features.names[idx - 1] == name


def test_committed_manifest_equals_regenerated_mapping():
    committed = MAPPING_FILE.read_text(encoding="ascii")
    regenerated = json.dumps(mapping_document(), indent=1, ensure_ascii=True) + "\n"
    assert committed == regenerated
    doc = json.loads(committed)
    assert doc["primary_variant"] == "X"
    assert doc["variants"]["X"]["indices_0based"] == [i - 1 for i in doc["variants"]["X"]["indices_1based"]]
