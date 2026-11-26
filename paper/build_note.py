"""Build the evidence note: figures from committed results -> Typst PDF -> PNG preview -> checks.

Run from the repository root with the paper environment (see paper/requirements-paper.txt):
    python paper/build_note.py
Outputs: paper/evidence_note.pdf (committed) and paper/evidence_note_preview.png (local, git-ignored).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typst
from pypdf import PdfReader

PAPER = Path(__file__).resolve().parent
SRC, PDF, PNG = PAPER / "evidence_note.typ", PAPER / "evidence_note.pdf", PAPER / "evidence_note_preview.png"
A4 = (595.2756, 841.8898)  # points


def main() -> int:
    subprocess.run([sys.executable, str(PAPER / "make_figures.py")], check=True)
    typst.compile(str(SRC), output=str(PDF))
    reader = PdfReader(str(PDF))
    n_pages = len(reader.pages)
    if n_pages == 1:
        typst.compile(str(SRC), output=str(PNG), format="png", ppi=300)
    box = reader.pages[0].mediabox
    size_mb = PDF.stat().st_size / 1e6
    a4 = abs(float(box.width) - A4[0]) < 0.5 and abs(float(box.height) - A4[1]) < 0.5
    meta = reader.metadata or {}
    print(f"pages={n_pages} a4={a4} size={size_mb:.3f} MB title={meta.get('/Title')!r} author={meta.get('/Author')!r}")
    ok = n_pages == 1 and a4 and size_mb < 10 and meta.get("/Author") == "Ikram Gafai"
    print("CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
