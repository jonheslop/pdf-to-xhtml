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
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Span:
    text: str
    italic: bool = False
    bold: bool = False


@dataclass
class Block:
    page: int
    size: float           # dominant (most-used) font size in the block; 0 for images
    bbox: tuple[float, float, float, float]
    lines: list[list[Span]] = field(default_factory=list)  # each line is a list of spans
    kind: str = "text"    # "text" or "image"

    def text(self, dehyphenate: bool = False) -> str:
        """Plain visible text (for analysis: inspect, length checks, sentence-end detection)."""
        return _render_paragraph(self.lines, dehyphenate=dehyphenate, html_mode=False)

    def html(
        self,
        dehyphenate: bool = False,
        emit_em: bool = True,
        emit_strong: bool = True,
        small_caps_pattern: "re.Pattern[str] | None" = None,
        small_caps_class: str = "ttl sc",
    ) -> str:
        """HTML-escaped text with optional <em>/<strong>/small-caps wrapping."""
        return _render_paragraph(
            self.lines,
            dehyphenate=dehyphenate,
            html_mode=True,
            emit_em=emit_em,
            emit_strong=emit_strong,
            small_caps_pattern=small_caps_pattern,
            small_caps_class=small_caps_class,
        )


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


_WHITESPACE_RUN = re.compile(r"\s+")


def _build_spans(raw_spans: list[dict]) -> list[Span]:
    """Build a list of Span objects from PyMuPDF's raw spans, collapsing
    whitespace within each span. Leading/trailing whitespace at the line
    edges is stripped by the caller."""
    out: list[Span] = []
    for s in raw_spans:
        text = _WHITESPACE_RUN.sub(" ", s["text"])
        if not text:
            continue
        flags = s.get("flags", 0)
        font = s.get("font", "")
        italic = _flags_italic(flags) or any(tag in font for tag in ("Italic", "Oblique"))
        bold = _flags_bold(flags) or any(tag in font for tag in ("Bold", "Black", "Heavy"))
        out.append(Span(text=text, italic=italic, bold=bold))
    return out


def _split_block(block: dict, page_index: int, indent_threshold: float) -> list[Block]:
    """Convert one PyMuPDF block into one or more Blocks.

    A single PyMuPDF block may contain multiple typographic paragraphs joined
    visually (no vertical space, just first-line indent). We split on lines
    that begin noticeably to the right of the block's body margin — those
    are first-line indents marking a new paragraph.
    """
    line_records = []
    for line in block["lines"]:
        raw_spans = line["spans"]
        if not raw_spans:
            continue
        spans = _build_spans(raw_spans)
        if not spans:
            continue
        # Strip leading whitespace from first span, trailing from last span.
        spans[0].text = spans[0].text.lstrip()
        spans[-1].text = spans[-1].text.rstrip()
        spans = [sp for sp in spans if sp.text]
        if not spans:
            continue
        sizes = [round(s["size"], 1) for s in raw_spans]
        line_records.append({
            "spans": spans,
            "bbox": tuple(line["bbox"]),
            "sizes": sizes,
        })

    if not line_records:
        return []

    # Body margin = the most common x0. Lines starting noticeably right of it
    # are paragraph-start indents.
    x0_mode = Counter(round(l["bbox"][0], 1) for l in line_records).most_common(1)[0][0]

    def line_size(rec):
        return Counter(rec["sizes"]).most_common(1)[0][0]

    paragraphs: list[list[dict]] = []
    current: list[dict] = []
    for rec in line_records:
        if current:
            is_indent_start = rec["bbox"][0] > x0_mode + indent_threshold
            # A different dominant size on this line vs the previous = different
            # typographic role (e.g., 10pt "CHAPTER 2" stacked above 24pt "DETAIL"
            # on a chapter opener). Split into separate paragraphs.
            is_size_change = abs(line_size(current[-1]) - line_size(rec)) > 1.0
            if is_indent_start or is_size_change:
                paragraphs.append(current)
                current = [rec]
                continue
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
                lines=[r["spans"] for r in para],
            )
        )
    return out


def _render_paragraph(
    lines: list[list[Span]],
    *,
    dehyphenate: bool = False,
    html_mode: bool = False,
    emit_em: bool = True,
    emit_strong: bool = True,
    small_caps_pattern: "re.Pattern[str] | None" = None,
    small_caps_class: str = "ttl sc",
) -> str:
    """Join a paragraph's lines into a single string.

    Lines ending in a hyphen glue to the next line without a space
    (preserving compound words like 'fragrant-smelling' that the typesetter
    broke across lines). If dehyphenate is True, the hyphen is dropped at
    line boundaries before an alpha continuation.

    With html_mode=False, returns plain visible text — useful for analysis
    (length checks, sentence-end detection). With html_mode=True, escapes
    HTML and wraps italic/bold spans in <em>/<strong>, coalescing adjacent
    same-tag spans so multi-line italics produce a single <em>…</em>.
    """
    if not lines:
        return ""

    # Per line-transition: decide separator (space or empty for hyphen-glue),
    # and whether the previous line's last span should drop its trailing hyphen.
    n = len(lines)
    separators = [""] * n
    drop_hyphen = [False] * n  # index of the line whose last span loses its trailing '-'
    for i in range(1, n):
        prev_line, cur_line = lines[i - 1], lines[i]
        if not prev_line or not cur_line:
            separators[i] = " "
            continue
        prev_last_text = prev_line[-1].text
        cur_first_text = cur_line[0].text
        if prev_last_text.endswith("-") and cur_first_text[:1].isalpha():
            separators[i] = ""
            if dehyphenate:
                drop_hyphen[i - 1] = True
        else:
            separators[i] = " "

    out: list[str] = []
    for line_idx, line in enumerate(lines):
        if separators[line_idx]:
            out.append(separators[line_idx])
        for span_idx, span in enumerate(line):
            text = span.text
            if (
                span_idx == len(line) - 1
                and drop_hyphen[line_idx]
                and text.endswith("-")
            ):
                text = text[:-1]
            if not text:
                continue
            if html_mode:
                text = html.escape(text, quote=False)
                if small_caps_pattern is not None:
                    cls_attr = html.escape(small_caps_class, quote=True)
                    text = small_caps_pattern.sub(
                        lambda m: f'<span class="{cls_attr}">{m.group(0)}</span>',
                        text,
                    )
                if emit_em and span.italic:
                    text = f"<em>{text}</em>"
                if emit_strong and span.bold:
                    text = f"<strong>{text}</strong>"
            out.append(text)

    result = "".join(out)

    if html_mode:
        # Coalesce adjacent same-tag spans (e.g. multi-line italics).
        result = re.sub(r"</em>(\s*)<em>", r"\1", result)
        result = re.sub(r"</strong>(\s*)<strong>", r"\1", result)

    return result


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
    # --drop-below/--drop-above only fire for *short* blocks (furniture-shaped)
    # so multi-paragraph small-font content (copyright, imprint) survives.
    def keep(b: Block) -> bool:
        if b.kind == "image":
            if args.no_images:
                return False
            w = b.bbox[2] - b.bbox[0]
            h = b.bbox[3] - b.bbox[1]
            return min(w, h) >= args.min_image_size
        looks_like_furniture = (
            args.max_furniture_chars == 0
            or len(b.text()) <= args.max_furniture_chars
        )
        if args.drop_below is not None and b.size < args.drop_below and looks_like_furniture:
            return False
        if args.drop_above is not None and b.size > args.drop_above and looks_like_furniture:
            return False
        return True

    filtered = [b for b in blocks if keep(b)]

    # 2. Merge paragraphs split across page breaks.
    if not args.no_merge:
        filtered = merge_paragraph_splits(filtered)

    # 3. Build size→(tag, class) mapping. --map SIZE=TAG or SIZE=TAG:CLASS.
    # Class None = use the default for the tag (e.g. --heading-class for h*).
    # Class "" = render with no class attribute.
    size_to_tag: dict[float, tuple[str, str | None]] = {}
    if args.body is not None:
        size_to_tag[args.body] = ("p", None)
    for pair in args.map or []:
        size_str, tag_spec = pair.split("=", 1)
        if ":" in tag_spec:
            tag, explicit_cls = tag_spec.split(":", 1)
        else:
            tag, explicit_cls = tag_spec, None
        size_to_tag[float(size_str)] = (tag, explicit_cls)

    # Always-on chapter detection: matches blocks that are exactly "CHAPTER N"
    # and emits them as <h2 class="chapter-number">CHAPTER N</h2>.
    chapter_marker = re.compile(r"^CHAPTER\s+\S+$")

    small_caps_pattern = re.compile(args.small_caps_pattern) if args.small_caps_pattern else None

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

        plain = b.text()
        if not plain:
            continue

        # Always-on chapter detection: "CHAPTER N" → <h2 class="chapter-number">CHAPTER N</h2>.
        m = chapter_marker.match(plain)
        if m and m.end() == len(plain):
            escaped = html.escape(m.group(0), quote=False)
            out_parts.append(
                f'<h2 class="{html.escape(args.chapter_number_class, quote=True)}">{escaped}</h2>'
            )
            prev_tag = "h2"
            prev_block = b
            continue

        tag, explicit_cls = size_to_tag.get(b.size, (args.default_tag, None))
        if explicit_cls is not None:
            cls = explicit_cls or None
        elif tag in themed_tags and args.heading_class:
            cls = args.heading_class
        elif tag == "p" and args.break_class:
            after_section_marker = prev_tag in non_continuation_tags
            big_gap = (
                args.break_gap > 0
                and prev_block is not None
                and prev_block.page == b.page
                and (b.bbox[1] - prev_block.bbox[3]) >= args.break_gap
            )
            cls = args.break_class if (after_section_marker or big_gap) else None
        else:
            cls = None
        # Headings stay clean (no <strong> — heading CSS already bolds them);
        # body keeps both emphasis tags.
        is_heading = tag in heading_tags
        rendered_text = b.html(
            dehyphenate=args.dehyphenate,
            emit_em=not args.no_emphasis,
            emit_strong=not args.no_emphasis and not is_heading,
            small_caps_pattern=small_caps_pattern if not is_heading else None,
            small_caps_class=args.small_caps_class,
        )
        out_parts.append(_render(tag, rendered_text, cls=cls))
        prev_tag = tag
        prev_block = b

    xhtml = _wrap_xhtml(_assemble_body(out_parts), title=args.title or Path(args.pdf).stem)
    out_path = Path(args.out)
    out_path.write_text(xhtml)
    print(f"Wrote {out_path} ({len(blocks)} blocks → {len(filtered)} kept → {len(out_parts)} elements)")
    return 0


def _render(tag: str, html_text: str, cls: str | None = None) -> str:
    """Wrap pre-escaped HTML content in an opening/closing tag pair.
    html_text is assumed to already be HTML-safe (output of Block.html())."""
    if cls:
        return f'<{tag} class="{html.escape(cls, quote=True)}">{html_text}</{tag}>'
    return f"<{tag}>{html_text}</{tag}>"


def _render_image(wrapper_class: str) -> str:
    cls = f' class="{html.escape(wrapper_class, quote=True)}"' if wrapper_class else ""
    return f'<p{cls}><img src="" alt=""/></p>'


_HEADING_PREFIXES = tuple(f"<h{n}" for n in range(1, 7))


def _assemble_body(parts: list[str]) -> str:
    """Join rendered elements with newlines, inserting a blank line before and
    after any heading (h1-h6) so the source is easier to scan."""
    if not parts:
        return ""
    out = [parts[0]]
    for i in range(1, len(parts)):
        part = parts[i]
        prev_was_heading = parts[i - 1].startswith(_HEADING_PREFIXES)
        is_heading = part.startswith(_HEADING_PREFIXES)
        out.append("\n\n    " if (is_heading or prev_was_heading) else "\n    ")
        out.append(part)
    return "".join(out)


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
        metavar="SIZE=TAG[:CLASS]",
        help="map a font size to a tag, optionally with a class. Repeatable. "
        "Examples: --map 24=h1, --map 24=h1:chapter-heading, "
        "--map 15=h2: (explicit no-class).",
    )
    pc.add_argument("--default-tag", default="p", help="tag for unmapped sizes (default: p)")
    pc.add_argument("--drop-below", type=float, help="drop blocks smaller than this (typically used to filter page numbers / running headers)")
    pc.add_argument("--drop-above", type=float, help="drop blocks larger than this")
    pc.add_argument(
        "--max-furniture-chars",
        type=int,
        default=30,
        help="only treat blocks as furniture (eligible for --drop-below/--drop-above) if their text is this short. Default: 30. Lets multi-paragraph small-font content (copyright/imprint) survive. Pass 0 to drop unconditionally by size.",
    )
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
        "--no-emphasis",
        action="store_true",
        help="skip <em>/<strong> wrapping around italic/bold spans (default: include them)",
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
    pc.add_argument(
        "--chapter-number-class",
        default="chapter-number",
        help='class on the <h2> that wraps a detected "CHAPTER N" line (default: chapter-number)',
    )
    pc.add_argument(
        "--small-caps-pattern",
        default="",
        help=(
            "regex matching runs of text to wrap as small caps. "
            r"OpenType smcp can't be detected from PDF metadata, so this is a heuristic. "
            r"Try '\b[A-Z]{4,}\b' (4+ letter caps catches LOVE/REMEMBER, skips AI/EU/FSC). "
            "Disabled by default; applies only to body, not headings. "
            "Watch for acronyms — you'll need to unwrap any false positives."
        ),
    )
    pc.add_argument(
        "--small-caps-class",
        default="ttl sc",
        help="class on the wrapping <span> for small-caps matches (default: 'ttl sc')",
    )
    pc.set_defaults(func=cmd_convert)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
