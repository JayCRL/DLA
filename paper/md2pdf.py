#!/usr/bin/env python3
"""Markdown -> PDF builder for the DLA paper draft.

Uses the Python ``markdown`` package for Markdown->HTML and ``fpdf2`` for
HTML->PDF. No LaTeX/TeX installation required.  Requires only:
    pip install markdown fpdf2

Usage:
    python3 paper/md2pdf.py [input.md] [output.pdf]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
DEFAULT_MD = ROOT / "DLA_paper_draft.md"
DEFAULT_PDF = ROOT / "DLA_paper_draft.pdf"
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

# macOS Arial Unicode supports Latin + CJK + math symbols like +/-/approx/arrow.
# On Linux, use e.g. /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf


def image_paths_absolute(html: str, base: Path) -> str:
    """Rewrite local figure paths to absolute paths and bound image widths."""
    def repl(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith("http"):
            return m.group(0)
        p = (base / src).resolve()
        return f'<img src="{p}" width="460">'
    return re.sub(r'<img\s+[^>]*?src="([^"]+)"[^>]*>', repl, html)


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    md_path = md_path.resolve()
    text = md_path.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "sane_lists"])
    # fpdf2's tiny HTML parser does not understand <hr>; remove horizontal rules.
    html = re.sub(r"<hr\s*/?>", "", html)
    # Replace symbols that fpdf2's HTML renderer may mishandle with ASCII forms.
    for a, b in [("→", "->"), ("←", "<-"), ("≈", "~"), ("±", "+/-"), ("×", "x"),
                 ("−", "-"), ("Δ", "delta"), ("φ", "phi"), ("α", "alpha"), ("Λ", "Lambda")]:
        html = html.replace(a, b)
    html = image_paths_absolute(html, md_path.parent)

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 15, 18)
    for style in ("", "B", "I", "BI"):
        pdf.add_font("PaperFont", style, FONT)
    pdf.set_font("PaperFont", size=10)
    pdf.add_page()

    pdf.write_html(html)

    pdf.output(str(pdf_path))
    print(f"PDF written: {pdf_path}")


if __name__ == "__main__":
    md = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_MD
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_PDF
    build_pdf(md, out)
