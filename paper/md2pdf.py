#!/usr/bin/env python3
"""Markdown -> PDF builder using pandoc + headless Chrome (no LaTeX, no blank-page bugs).

Requirements:
  - pandoc on PATH
  - Google Chrome in /Applications/Google Chrome.app

Usage:
    python3 paper/md2pdf.py [input.md] [output.pdf]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MD = ROOT / "DLA_paper_draft.md"
DEFAULT_PDF = ROOT / "DLA_paper_draft.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    md_path = md_path.resolve()
    pdf_path = pdf_path.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        html_path = tmp / "paper.html"
        # pandoc standalone HTML; copy figures next to it so Chrome can load them.
        run(["pandoc", str(md_path), "-f", "markdown", "-t", "html5", "-s",
             "-o", str(html_path), "--metadata", "title=DLA Draft"])
        fig_src = md_path.parent / "figures"
        if fig_src.exists():
            shutil.copytree(fig_src, tmp / "figures")
        run([CHROME, "--headless", "--disable-gpu", "--no-sandbox",
             "--print-to-pdf=" + str(pdf_path),
             "--no-margins", "file://" + str(html_path)])
    print(f"PDF written: {pdf_path}")


if __name__ == "__main__":
    md = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_MD
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_PDF
    build_pdf(md, out)
