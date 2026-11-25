"""Map the paper's 100 published feature names (Supplementary Table S2) to original UCI HAR column indices.

Ninety-one names occur exactly once in features.txt. Nine `bandsEnergy` names occur three times (one block per
sensor axis), and the supplement gives no index or axis for them. The pre-registered primary rule picks the first
occurrence ("X"). The exploratory variants "Y" and "Z" move all nine together to the second or third occurrence.
This is a declared convention, not a claim about which columns the authors used.
Columns are always returned in ascending original-index order.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

VARIANT_POSITION = {"X": 0, "Y": 1, "Z": 2}


@dataclass(frozen=True)
class AmbiguousFeature:
    name: str
    candidates: tuple[int, ...]   # 1-based original indices, in file order (block 1, 2, 3)
    chosen: int                   # 1-based index selected under the variant


@dataclass(frozen=True)
class CompactMapping:
    variant: str
    indices_1based: tuple[int, ...]     # ascending
    names: tuple[str, ...]              # names aligned with indices_1based
    n_unique: int
    ambiguous: tuple[AmbiguousFeature, ...]

    @property
    def indices_0based(self) -> tuple[int, ...]:
        return tuple(i - 1 for i in self.indices_1based)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "column_order": "ascending_original_index",
            "n_selected": len(self.indices_1based),
            "n_unique_name_matches": self.n_unique,
            "n_ambiguous_names": len(self.ambiguous),
            "indices_1based": list(self.indices_1based),
            "indices_0based": list(self.indices_0based),
            "names": list(self.names),
            "ambiguous": [
                {"name": a.name, "candidates_1based": list(a.candidates), "chosen_1based": a.chosen}
                for a in self.ambiguous
            ],
        }


def read_s2_feature_names(path: Path) -> list[str]:
    """Return the 100 names in the order published in S2 (column 'Original_UCI_Feature')."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    names = [r["Original_UCI_Feature"] for r in rows if r.get("Original_UCI_Feature")]
    if len(names) != len(set(names)):
        raise ValueError("S2 lists a feature name more than once")
    return names


def build_mapping(selected_names: Sequence[str], feature_names: Sequence[str], variant: str = "X") -> CompactMapping:
    if variant not in VARIANT_POSITION:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANT_POSITION)}")
    positions: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names, start=1):
        positions.setdefault(name, []).append(i)

    chosen: dict[int, str] = {}
    ambiguous: list[AmbiguousFeature] = []
    n_unique = 0
    missing = [n for n in selected_names if n not in positions]
    if missing:
        raise KeyError(f"selected names not found in features.txt: {missing}")
    for name in selected_names:
        cands = positions[name]
        if len(cands) == 1:
            idx = cands[0]
            n_unique += 1
        elif len(cands) == 3:
            idx = cands[VARIANT_POSITION[variant]]
            ambiguous.append(AmbiguousFeature(name, tuple(cands), idx))
        else:
            raise ValueError(f"{name!r} occurs {len(cands)} times; only 1 or 3 occurrences are handled")
        if idx in chosen:
            raise ValueError(f"index {idx} selected twice ({chosen[idx]!r}, {name!r})")
        chosen[idx] = name

    order = sorted(chosen)
    ambiguous.sort(key=lambda a: a.candidates[0])
    return CompactMapping(
        variant=variant,
        indices_1based=tuple(order),
        names=tuple(chosen[i] for i in order),
        n_unique=n_unique,
        ambiguous=tuple(ambiguous),
    )
