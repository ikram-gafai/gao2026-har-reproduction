"""Acquire, verify and extract the public UCI HAR dataset.

Policy:
1. A local archive (optional) or each configured source, in order, is accepted only if its SHA-256 equals the
   expected hash for its kind ("outer" official ZIP or "inner" 'UCI HAR Dataset.zip').
2. The inner archive is SHA-256-verified, extracted with path-traversal protection (macOS metadata skipped),
   and the key extracted files are verified against expected per-file hashes.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

USER_AGENT = "har-repro/0.1 (research reproduction; python-urllib)"
CHUNK = 1 << 16


class IntegrityError(RuntimeError):
    """Raised when a file does not match its expected checksum or structure."""


@dataclass(frozen=True)
class AcquiredArchive:
    path: Path
    kind: str          # "outer" or "inner"
    sha256: str
    source: str        # source name or "local:<path>"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def classify_archive(sha: str, expected: Mapping[str, Any]) -> str | None:
    if sha == expected["outer_sha256"]:
        return "outer"
    if sha == expected["inner_sha256"]:
        return "inner"
    return None


def http_download(url: str, dest: Path, timeout: float = 60.0, log: Callable[[str], None] = print) -> None:
    """Stream url to dest via a .part file (renamed only on success)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    received, next_report = 0, 5 << 20
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as out:
        for block in iter(lambda: resp.read(CHUNK), b""):
            out.write(block)
            received += len(block)
            if received >= next_report:
                log(f"  ... {received / 1e6:.1f} MB")
                next_report += 5 << 20
    part.replace(dest)


def acquire_archive(
    sources_cfg: Mapping[str, Any],
    raw_dir: Path,
    local_archive: Path | None = None,
    downloader: Callable[[str, Path], None] | None = None,
    log: Callable[[str], None] = print,
) -> AcquiredArchive:
    """Return a checksum-verified archive in raw_dir, from local_archive or the configured sources."""
    expected = sources_cfg["expected"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Reuse an already-verified archive.
    for kind, fname in (("outer", "uci_har_outer.zip"), ("inner", "uci_har_inner.zip")):
        cached = raw_dir / fname
        if cached.exists():
            sha = sha256_file(cached)
            if sha == expected[f"{kind}_sha256"]:
                log(f"Using cached verified archive {cached.name} ({sha[:12]}...)")
                return AcquiredArchive(cached, kind, sha, "cache")
            log(f"Cached {cached.name} has wrong checksum; ignoring it")

    if local_archive is not None:
        local_archive = Path(local_archive)
        sha = sha256_file(local_archive)
        kind = classify_archive(sha, expected)
        if kind is None:
            raise IntegrityError(f"local archive {local_archive} has unexpected SHA-256 {sha}")
        dest = raw_dir / f"uci_har_{kind}.zip"
        shutil.copyfile(local_archive, dest)
        if sha256_file(dest) != sha:
            raise IntegrityError("copy of local archive does not match its source checksum")
        log(f"Accepted local {kind} archive {local_archive.name} (SHA-256 {sha})")
        return AcquiredArchive(dest, kind, sha, f"local:{local_archive}")

    downloader = downloader or (lambda url, dest: http_download(url, dest, log=log))
    errors = []
    for src in sources_cfg["sources"]:
        kind = src["kind"]
        dest = raw_dir / f"uci_har_{kind}.zip"
        log(f"Trying {src['name']}: {src['url']}")
        try:
            downloader(src["url"], dest)
        except Exception as exc:  # network failures fall through to the next source
            errors.append(f"{src['name']}: {exc!r}")
            log(f"  failed: {exc!r}")
            continue
        sha = sha256_file(dest)
        if sha != expected[f"{kind}_sha256"]:
            errors.append(f"{src['name']}: checksum mismatch {sha}")
            log(f"  REJECTED: SHA-256 {sha} != expected {expected[f'{kind}_sha256']}")
            dest.unlink(missing_ok=True)
            continue
        log(f"  verified SHA-256 {sha}")
        return AcquiredArchive(dest, kind, sha, src["name"])
    raise IntegrityError("no source produced a verified archive:\n  " + "\n  ".join(errors))


def _inner_archive_bytes(archive: AcquiredArchive, expected: Mapping[str, Any]) -> bytes:
    data = archive.path.read_bytes()
    if archive.kind == "outer":
        with zipfile.ZipFile(io.BytesIO(data)) as outer:
            bad = outer.testzip()
            if bad is not None:
                raise IntegrityError(f"outer archive member failed CRC: {bad}")
            data = outer.read(expected["inner_member_name"])
    if sha256_bytes(data) != expected["inner_sha256"]:
        raise IntegrityError("inner 'UCI HAR Dataset.zip' SHA-256 mismatch")
    return data


def _is_metadata(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (not parts) or parts[0] == "__MACOSX" or parts[-1] == ".DS_Store"


def extract_dataset(archive: AcquiredArchive, sources_cfg: Mapping[str, Any], extracted_dir: Path,
                    log: Callable[[str], None] = print) -> Path:
    """Extract the verified inner archive and verify the expected per-file hashes. Returns the dataset root."""
    expected = sources_cfg["expected"]
    inner = _inner_archive_bytes(archive, expected)
    root = (extracted_dir / expected["dataset_dirname"]).resolve()
    extracted_dir.mkdir(parents=True, exist_ok=True)
    base = extracted_dir.resolve()
    with zipfile.ZipFile(io.BytesIO(inner)) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise IntegrityError(f"inner archive member failed CRC: {bad}")
        for info in zf.infolist():
            if _is_metadata(info.filename):
                continue
            target = (base / info.filename).resolve()
            if base != target and base not in target.parents:
                raise IntegrityError(f"unsafe path in archive: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    verify_extracted(root, expected["files"])
    log(f"Extracted and verified {len(expected['files'])} key files under {root}")
    return root


def verify_extracted(root: Path, expected_files: Mapping[str, str]) -> None:
    mismatches = []
    for rel, want in expected_files.items():
        path = root / rel
        if not path.exists():
            mismatches.append(f"{rel}: missing")
            continue
        got = sha256_file(path)
        if got != want:
            mismatches.append(f"{rel}: {got} != {want}")
    if mismatches:
        raise IntegrityError("extracted file verification failed:\n  " + "\n  ".join(mismatches))


def prepare_dataset(sources_cfg: Mapping[str, Any], raw_dir: Path, extracted_dir: Path,
                    local_archive: Path | None = None, log: Callable[[str], None] = print) -> Path:
    archive = acquire_archive(sources_cfg, raw_dir, local_archive=local_archive, log=log)
    return extract_dataset(archive, sources_cfg, extracted_dir, log=log)


if __name__ == "__main__":  # pragma: no cover
    sys.exit("Use: python -m har_repro.cli prepare-data")
