"""Acquisition and integrity logic on synthetic archives (no network)."""

import hashlib
import io
import zipfile

import pytest

from har_repro.data.acquire import (IntegrityError, acquire_archive, extract_dataset, sha256_bytes,
                                    sha256_file, verify_extracted)

FILES = {"features.txt": b"1 a\n2 b\n", "train/X_train.txt": b" 1.0 2.0\n"}


def _inner_zip(files=FILES, extra=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in files.items():
            z.writestr(f"UCI HAR Dataset/{rel}", data)
        z.writestr("__MACOSX/UCI HAR Dataset/._features.txt", b"meta")
        z.writestr("UCI HAR Dataset/.DS_Store", b"meta")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def _outer_zip(inner: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("UCI HAR Dataset.names", b"names")
        z.writestr("UCI HAR Dataset.zip", inner)
    return buf.getvalue()


def _cfg(inner: bytes, outer: bytes, sources=None):
    return {
        "expected": {
            "outer_sha256": sha256_bytes(outer), "inner_sha256": sha256_bytes(inner),
            "inner_member_name": "UCI HAR Dataset.zip", "dataset_dirname": "UCI HAR Dataset",
            "files": {rel: hashlib.sha256(d).hexdigest() for rel, d in FILES.items()},
        },
        "sources": sources or [
            {"name": "s-inner", "kind": "inner", "url": "u1"},
            {"name": "s-outer", "kind": "outer", "url": "u2"},
        ],
    }


def test_local_outer_archive_is_verified_and_extracted(tmp_path):
    inner = _inner_zip()
    outer = _outer_zip(inner)
    local = tmp_path / "official.zip"
    local.write_bytes(outer)
    cfg = _cfg(inner, outer)
    arch = acquire_archive(cfg, tmp_path / "raw", local_archive=local, log=lambda *_: None)
    assert arch.kind == "outer" and arch.sha256 == sha256_bytes(outer)
    root = extract_dataset(arch, cfg, tmp_path / "ext", log=lambda *_: None)
    assert (root / "features.txt").read_bytes() == FILES["features.txt"]
    assert not (tmp_path / "ext" / "__MACOSX").exists()
    assert not (root / ".DS_Store").exists()


def test_local_archive_with_wrong_checksum_is_rejected(tmp_path):
    inner = _inner_zip()
    cfg = _cfg(inner, _outer_zip(inner))
    bad = tmp_path / "tampered.zip"
    bad.write_bytes(_outer_zip(_inner_zip({**FILES, "features.txt": b"1 tampered\n"})))
    with pytest.raises(IntegrityError):
        acquire_archive(cfg, tmp_path / "raw", local_archive=bad, log=lambda *_: None)


def test_sources_tried_in_order_and_bad_checksum_falls_through(tmp_path):
    inner = _inner_zip()
    outer = _outer_zip(inner)
    cfg = _cfg(inner, outer)
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        if url == "u1":
            dest.write_bytes(b"corrupted bytes")   # wrong checksum -> rejected
        else:
            dest.write_bytes(outer)

    arch = acquire_archive(cfg, tmp_path / "raw", downloader=fake_download, log=lambda *_: None)
    assert calls == ["u1", "u2"]
    assert arch.source == "s-outer" and arch.kind == "outer"
    assert not (tmp_path / "raw" / "uci_har_inner.zip").exists()


def test_network_failure_on_all_sources_raises(tmp_path):
    inner = _inner_zip()
    cfg = _cfg(inner, _outer_zip(inner))

    def failing(url, dest):
        raise OSError("offline")

    with pytest.raises(IntegrityError):
        acquire_archive(cfg, tmp_path / "raw", downloader=failing, log=lambda *_: None)


def test_cached_verified_archive_is_reused_without_download(tmp_path):
    inner = _inner_zip()
    cfg = _cfg(inner, _outer_zip(inner))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "uci_har_inner.zip").write_bytes(inner)

    def must_not_download(url, dest):
        raise AssertionError("download attempted despite verified cache")

    arch = acquire_archive(cfg, raw, downloader=must_not_download, log=lambda *_: None)
    assert arch.source == "cache" and arch.kind == "inner"


def test_path_traversal_member_is_refused(tmp_path):
    inner = _inner_zip(extra={"../escape.txt": b"x"})
    outer = _outer_zip(inner)
    cfg = _cfg(inner, outer)
    local = tmp_path / "o.zip"
    local.write_bytes(outer)
    arch = acquire_archive(cfg, tmp_path / "raw", local_archive=local, log=lambda *_: None)
    with pytest.raises(IntegrityError):
        extract_dataset(arch, cfg, tmp_path / "ext", log=lambda *_: None)
    assert not (tmp_path / "escape.txt").exists()


def test_verify_extracted_detects_modified_file(tmp_path):
    root = tmp_path / "UCI HAR Dataset"
    (root / "train").mkdir(parents=True)
    for rel, data in FILES.items():
        (root / rel).write_bytes(data)
    expected = {rel: hashlib.sha256(d).hexdigest() for rel, d in FILES.items()}
    verify_extracted(root, expected)
    (root / "features.txt").write_bytes(b"1 a\n2 changed\n")
    with pytest.raises(IntegrityError):
        verify_extracted(root, expected)
    assert sha256_file(root / "train/X_train.txt") == expected["train/X_train.txt"]
