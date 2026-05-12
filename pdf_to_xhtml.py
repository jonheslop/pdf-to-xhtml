"""Convert a print-PDF into clean XHTML for an EPUB workflow.

Two subcommands:

  inspect  Print a font-size histogram and a few sample blocks at each size,
           so you can decide which size maps to which tag for THIS book.

  convert  Extract text as blocks, drop page furniture, merge paragraphs
           that span page breaks, map font sizes to tags per your --map
           flags, and emit one XHTML file.

Why blocks, not raw text:
  PyMuPDF's block-level extraction already groups each paragraph into a
  single block with one bbox. That means the "paragraph reflow" problem
  (designer line-breaks that don't belong in flowing text) is solved for
  free — we just join the lines inside a block with a space. Unicode
  characters (em dashes, smart quotes) come through untouched.

Typical workflow:
    python pdf_to_xhtml.py inspect book.pdf
    python pdf_to_xhtml.py convert book.pdf \\
        --body 10 --map 24=h1 --map 15=h2 --map 12.5=h3 \\
        --drop-below 9 --out book.xhtml
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Block:
    page: int
    size: float           # dominant (most-used) font size in the block; 0 for images
    bbox: tuple[float, float, float, float]
    lines: list[str] = field(default_factory=list)
    bold: bool = False
    italic: bool = False
    kind: str = "text"    # "text" or "image"

    def text(self, dehyphenate: bool = False) -> str:
        return _join_lines(self.lines, dehyphenate=dehyphenate)


def _flags_bold(flags: int) -> bool:
    return bool(flags & 16)


def _flags_italic(flags: int) -> bool:
    return bool(flags & 2)


def read_blocks(
    pdf_path: Path,
    page_range: tuple[int, int] | None = None,
    indent_threshold: float = 3.0,
) -> list[Block]:
    doc = fitz.open(pdf_path)
    if page_range:
        start, end = page_range
        doc.select(list(range(start - 1, end)))

    blocks: list[Block] = []
    for page_index, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            btype = block.get("type")
            if btype == 0:
                blocks.extend(_split_block(block, page_index, indent_threshold))
            elif btype == 1:
                blocks.append(
                    Block(
                        page=page_index,
                        size=0,
                        bbox=tuple(block["bbox"]),
                        kind="image",
                    )
                )
    return blocks


def _split_block(block: dict, page_index: int, indent_threshold: float) -> list[Block]:
    """Convert one PyMuPDF block into one or more Blocks.

    A single PyMuPDF block may contain multiple typographic paragraphs joined
    visually (no vertical space, just first-line indent). We split on lines
    that begin noticeably to the right of the block's body margin — those
    are first-line indents marking a new paragraph.
    """
    line_records = []
    for line in block["lines"]:
        spans = line["spans"]
        if not spans:
            continue
        text = "".join(s["text"] for s in spans).strip()
        if not text:
            continue
        sizes = [round(s["size"], 1) for s in spans]
        bold = any(
            _flags_bold(s.get("flags", 0))
            or any(tag in s.get("font", "") for tag in ("Bold", "Black", "Heavy"))
            for s in spans
        )
        italic = any(
            _flags_italic(s.get("flags", 0))
            or any(tag in s.get("font", "") for tag in ("Italic", "Oblique"))
            for s in spans
        )
        line_records.append({
            "text": text,
            "bbox": tuple(line["bbox"]),
            "sizes": sizes,
            "bold": bold,
            "italic": italic,
        })

    if not line_records:
        return []

    # Body margin = the most common x0. Lines starting noticeably right of it
    # are paragraph-start indents.
    x0_mode = Counter(round(l["bbox"][0], 1) for l in line_records).most_common(1)[0][0]

    paragraphs: list[list[dict]] = []
    current: list[dict] = []
    for rec in line_records:
        is_indent_start = current and rec["bbox"][0] > x0_mode + indent_threshold
        if is_indent_start:
            paragraphs.append(current)
            current = [rec]
        else:
            current.append(rec)
    if current:
        paragraphs.append(current)

    out: list[Block] = []
    for para in paragraphs:
        all_sizes = [s for rec in para for s in rec["sizes"]]
        bbox = (
            min(r["bbox"][0] for r in para),
            min(r["bbox"][1] for r in para),
            max(r["bbox"][2] for r in para),
            max(r["bbox"][3] for r in para),
        )
        out.append(
            Block(
                page=page_index,
                size=Counter(all_sizes).most_common(1)[0][0],
                bbox=bbox,
                lines=[r["text"] for r in para],
                bold=any(r["bold"] for r in para),
                italic=any(r["italic"] for r in para),
            )
        )
    return out


def _join_lines(lines: list[str], dehyphenate: bool = False) -> str:
    """Join lines into a single paragraph.

    Lines ending in a hyphen are glued to the next line without a space
    (so 'fragrant-' + 'smelling' → 'fragrant-smelling'). If dehyphenate
    is True, the hyphen itself is also dropped — only useful for books
    whose typesetter breaks single words across lines.
    """
    if not lines:
        return ""
    parts = [lines[0]]
    for line in lines[1:]:
        prev = parts[-1]
        if prev.endswith("-") and line[:1].isalpha():
            parts[-1] = (prev[:-1] if dehyphenate else prev) + line
        else:
            parts.append(line)
    return " ".join(parts)


# Sentence-terminal punctuation. A block ending in any of these (optionally
# followed by a closing quote) is considered a finished paragraph.
_TERMINAL = set(".!?…")
_CLOSING = set('"\'’”»')  # straight + smart right quotes, »


def _ends_paragraph(text: str) -> bool:
    s = text.rstrip()
    if not s:
        return True
    last = s[-1]
    if last in _TERMINAL:
        return True
    if last in _CLOSING and len(s) >= 2 and s[-2] in _TERMINAL:
        return True
    return False


def _starts_continuation(text: str) -> bool:
    s = text.lstrip()
    if not s:
        return False
    first = s[0]
    return first.islower() or first in ",;:)]}"


def merge_paragraph_splits(blocks: list[Block]) -> list[Block]:
    """Merge consecutive same-size blocks where the first ends mid-sentence
    and the second starts with a lowercase letter (typical sign of a
    paragraph that spilled across a page break)."""
    out: list[Block] = []
    for b in blocks:
        if (
            out
            and out[-1].size == b.size
            and not _ends_paragraph(out[-1].text())
            and _starts_continuation(b.text())
        ):
            prev = out[-1]
            prev.lines = prev.lines + b.lines
            prev.bold = prev.bold or b.bold
            prev.italic = prev.italic or b.italic
        else:
            out.append(b)
    return out


# ---------- inspect ----------

def cmd_inspect(args: argparse.Namespace) -> int:
    blocks = read_blocks(Path(args.pdf), _parse_pages(args.pages))

    by_size: dict[float, list[Block]] = defaultdict(list)
    for b in blocks:
        by_size[b.size].append(b)

    char_counts = {sz: sum(len(b.text()) for b in bs) for sz, bs in by_size.items()}
    print(f"\n{Path(args.pdf).name}: {len(blocks)} blocks")
    print(f"{'size':>6}  {'blocks':>6}  {'chars':>7}  sample")
    print("-" * 78)
    for sz in sorted(by_size, key=lambda s: -char_counts[s]):
        bs = by_size[sz]
        sample = bs[0].text()
        if len(sample) > 50:
            sample = sample[:47] + "…"
        print(f"{sz:>6.1f}  {len(bs):>6}  {char_counts[sz]:>7}  {sample!r}")

    if args.show:
        for sz in sorted(by_size, key=lambda s: -s):
            bs = by_size[sz]
            print(f"\n===== {sz}pt — {len(bs)} block(s) =====")
            for b in bs[: args.show]:
                preview = b.text()
                if len(preview) > 200:
                    preview = preview[:197] + "…"
                print(f"  [p.{b.page} bbox={_fmt_bbox(b.bbox)}]")
                print(f"  {preview!r}")
    return 0


def _fmt_bbox(bbox: tuple[float, float, float, float]) -> str:
    return f"({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"


# ---------- convert ----------

def cmd_convert(args: argparse.Namespace) -> int:
    blocks = read_blocks(Path(args.pdf), _parse_pages(args.pages))

    # 1. Filter page furniture (small/large outliers) and decorative images.
    def keep(b: Block) -> bool:
        if b.kind == "image":
            if args.no_images:
                return False
            w = b.bbox[2] - b.bbox[0]
            h = b.bbox[3] - b.bbox[1]
            return min(w, h) >= args.min_image_size
        if args.drop_below is not None and b.size < args.drop_below:
            return False
        if args.drop_above is not None and b.size > args.drop_above:
            return False
        return True

    filtered = [b for b in blocks if keep(b)]

    # 2. Merge paragraphs split across page breaks.
    if not args.no_merge:
        filtered = merge_paragraph_splits(filtered)

    # 3. Build size→tag mapping.
    size_to_tag: dict[float, str] = {}
    if args.body is not None:
        size_to_tag[args.body] = "p"
    for pair in args.map or []:
        size_str, tag = pair.split("=", 1)
        size_to_tag[float(size_str)] = tag

    # 4. Render. Track previous element so we can mark first-of-section <p>s.
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    themed_tags = {"h1", "h2", "h3"}
    non_continuation_tags = heading_tags | {"image"}
    out_parts: list[str] = []
    prev_tag: str | None = None
    prev_block: Block | None = None
    for b in filtered:
        if b.kind == "image":
            out_parts.append(_render_image(args.image_class))
            prev_tag = "image"
            prev_block = b
            continue

        text = b.text(dehyphenate=args.dehyphenate)
        if not text:
            continue
        tag = size_to_tag.get(b.size, args.default_tag)
        cls: str | None = None
        if tag in themed_tags and args.heading_class:
            cls = args.heading_class
        elif tag == "p" and args.break_class:
            after_section_marker = prev_tag in non_continuation_tags
            big_gap = (
                args.break_gap > 0
                and prev_block is not None
                and prev_block.page == b.page
                and (b.bbox[1] - prev_block.bbox[3]) >= args.break_gap
            )
            if after_section_marker or big_gap:
                cls = args.break_class
        out_parts.append(_render(tag, text, cls=cls))
        prev_tag = tag
        prev_block = b

    xhtml = _wrap_xhtml("\n    ".join(out_parts), title=args.title or Path(args.pdf).stem)
    out_path = Path(args.out)
    out_path.write_text(xhtml)
    print(f"Wrote {out_path} ({len(blocks)} blocks → {len(filtered)} kept → {len(out_parts)} elements)")
    return 0


def _render(tag: str, text: str, cls: str | None = None) -> str:
    escaped = html.escape(text, quote=False)
    if cls:
        return f'<{tag} class="{html.escape(cls, quote=True)}">{escaped}</{tag}>'
    return f"<{tag}>{escaped}</{tag}>"


def _render_image(wrapper_class: str) -> str:
    cls = f' class="{html.escape(wrapper_class, quote=True)}"' if wrapper_class else ""
    return f'<p{cls}><img src="" alt=""/></p>'


def _wrap_xhtml(body: str, title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">\n'
        "  <head>\n"
        f"    <title>{html.escape(title)}</title>\n"
        '    <meta http-equiv="Default-Style" content="application/xhtml+xml; charset=UTF-8"/>\n'
        '    <link rel="stylesheet" type="text/css" href="styles.css"/>\n'
        "  </head>\n"
        "  <body>\n"
        f"    {body}\n"
        "  </body>\n"
        "</html>\n"
    )


# ---------- shared ----------

def _parse_pages(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    a, b = spec.split("-", 1)
    return int(a), int(b)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pdf-to-xhtml",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="show font-size histogram + sample blocks")
    pi.add_argument("pdf")
    pi.add_argument("--pages", help="page range, e.g. 20-25 (1-indexed, inclusive)")
    pi.add_argument("--show", type=int, default=2, help="sample blocks to print per size (default: 2)")
    pi.set_defaults(func=cmd_inspect)

    pc = sub.add_parser("convert", help="extract → XHTML")
    pc.add_argument("pdf")
    pc.add_argument("--out", required=True, help="output xhtml path")
    pc.add_argument("--pages", help="page range, e.g. 20-25")
    pc.add_argument("--title", help="HTML <title> (default: pdf filename stem)")
    pc.add_argument("--body", type=float, help="font size to treat as body <p>")
    pc.add_argument(
        "--map",
        action="append",
        metavar="SIZE=TAG",
        help="map a font size to a tag, e.g. --map 24=h1. Repeatable.",
    )
    pc.add_argument("--default-tag", default="p", help="tag for unmapped sizes (default: p)")
    pc.add_argument("--drop-below", type=float, help="drop blocks smaller than this (filters page numbers/captions)")
    pc.add_argument("--drop-above", type=float, help="drop blocks larger than this")
    pc.add_argument(
        "--no-merge",
        action="store_true",
        help="disable cross-page paragraph merging (on by default)",
    )
    pc.add_argument(
        "--dehyphenate",
        action="store_true",
        help="drop end-of-line hyphens (off by default; only useful if the typesetter breaks single words across lines)",
    )
    pc.add_argument(
        "--heading-class",
        default="theme",
        help="class to add to h1/h2/h3 (default: theme; pass '' to disable)",
    )
    pc.add_argument(
        "--break-class",
        default="break",
        help="class to add to first <p> after a heading or a large vertical gap (default: break; pass '' to disable)",
    )
    pc.add_argument(
        "--break-gap",
        type=float,
        default=0.0,
        help="vertical gap (pt) above a <p> that triggers --break-class; 0 disables (default). Try ~1.5× body size.",
    )
    pc.add_argument(
        "--no-images",
        action="store_true",
        help="skip image blocks entirely (default: emit <p class=image><img src='' alt=''/></p>)",
    )
    pc.add_argument(
        "--min-image-size",
        type=float,
        default=0.0,
        help="drop images whose smaller dimension is below this many pt (default: 0). Try ~100 to skip logos and decorative bits.",
    )
    pc.add_argument(
        "--image-class",
        default="image",
        help="class on the <p> wrapper around <img> (default: image; pass '' to disable)",
    )
    pc.set_defaults(func=cmd_convert)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
