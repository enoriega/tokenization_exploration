from typing import Generator

from lxml import etree
import re, copy

# ---------------------------------------------------------------------------
# JATS / .nxml full-text extraction
# ---------------------------------------------------------------------------
#
# Extract a clean, plain-text rendering of a PMC article suitable for training
# an encoder model. JATS inline markup (<italic>, <sup>, <xref>, ...) is
# flattened to its text content. Tags are matched by *local name* so the code
# is agnostic to namespace prefixes / default namespaces across PMC variants.

# Wrapper elements whose <label> + <caption> we keep, but whose body content
# (the figure graphic, the table grid, etc.) we deliberately skip.
_CAPTION_TAGS = {"fig", "table-wrap", "boxed-text", "supplementary-material"}


def _local(elem) -> str | None:
    """Return the namespace-stripped tag name, or None for comments/PIs."""
    tag = elem.tag
    if not isinstance(tag, str):
        return None
    return tag.rsplit("}", 1)[-1]


def _child(elem, name):
    """First direct child with the given local name, or None."""
    for child in elem:
        if _local(child) == name:
            return child
    return None


def _iter_local(elem, name) -> Generator:
    """Yield every descendant (and self) with the given local name."""
    for e in elem.iter():
        if _local(e) == name:
            yield e


def _find_local(elem, name):
    """First descendant (or self) with the given local name, or None."""
    return next(_iter_local(elem, name), None)


def _normalize(text: str) -> str:
    """Collapse whitespace and tidy artifacts left behind after stripping refs."""
    text = re.sub(r"\s+", " ", text).strip()
    # Stripping <xref ref-type="bibr"> leaves empty citation brackets like
    # "[,,]" or "[]"; drop them along with any preceding space.
    text = re.sub(r"\s*\[[\s,;–-]*\]", "", text)
    text = re.sub(r"\s*\(\s*[,;–-]*\s*\)", "", text)
    # Tidy spacing before punctuation and any doubled-up punctuation that the
    # bracket removal may have produced (e.g. "spp. ." -> "spp.").
    text = re.sub(r"\s+([.,;:)\]])", r"\1", text)
    text = re.sub(r"([.,;])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _drop_keep_tail(elem) -> None:
    """Remove an element from its parent but preserve its tail text."""
    parent = elem.getparent()
    if parent is None:
        return
    if elem.tail:
        prev = elem.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + elem.tail
        else:
            parent.text = (parent.text or "") + elem.tail
    parent.remove(elem)


def _flatten(elem, strip_bibr: bool = False) -> str:
    """Flatten an element to normalized plain text.

    When ``strip_bibr`` is set, inline bibliographic citation markers
    (``<xref ref-type="bibr">``, the "[1,2]" in running text) are removed
    first; other cross-references (e.g. "Table 1", "Figure 2") are kept.
    """
    if elem is None:
        return ""
    if strip_bibr:
        elem = copy.deepcopy(elem)
        for xref in list(_iter_local(elem, "xref")):
            if xref.get("ref-type") == "bibr":
                _drop_keep_tail(xref)
    return _normalize("".join(elem.itertext()))


def _format_authors(front) -> str:
    """Comma-separated "Given Surname" list of the article's authors."""
    if front is None:
        return ""
    authors = []
    for contrib in _iter_local(front, "contrib"):
        if contrib.get("contrib-type") != "author":
            continue
        name = _child(contrib, "name")
        if name is None:
            continue
        given = _flatten(_child(name, "given-names"))
        surname = _flatten(_child(name, "surname"))
        full = " ".join(part for part in (given, surname) if part)
        if full:
            authors.append(full)
    return ", ".join(authors)


def _walk_body(elem, parts: list[str]) -> None:
    """Recursively emit section titles, paragraphs and captions in reading order."""
    for child in elem:
        tag = _local(child)
        if tag is None:
            continue
        if tag == "sec":
            title = _child(child, "title")
            if title is not None:
                text = _flatten(title)
                if text:
                    parts.append(text)
            _walk_body(child, parts)
        elif tag == "p":
            text = _flatten(child, strip_bibr=True)
            if text:
                parts.append(text)
        elif tag in _CAPTION_TAGS:
            label = _flatten(_child(child, "label"))
            caption = _flatten(_child(child, "caption"), strip_bibr=True)
            text = " ".join(piece for piece in (label, caption) if piece)
            if text:
                parts.append(text)
        elif tag == "title":
            # Already emitted by the parent <sec> handler above.
            continue
        else:
            # Generic container (list, disp-quote, ...): recurse for nested text.
            _walk_body(child, parts)


def extract_text_from_nxml(xml: str) -> str:
    """Extract plain text from a PMC/JATS article for encoder pre-training.

    Elements are emitted in the order they appear in the XML, preserving the
    narrative flow: the article's title, authors and abstract, followed by the
    body (section titles, paragraphs and figure/table captions in reading
    order), with each element on its own line. The reference list / bibliography
    is intentionally dropped. Inline markup is flattened and in-text citation
    markers (``[1,2]``) are stripped from the prose. Returns an empty string if
    the XML cannot be parsed at all.
    """
    if not xml:
        return ""
    # Encode to bytes so an XML encoding declaration doesn't trip lxml, and use
    # a recovering parser so the occasional malformed article still yields text.
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False)
    try:
        root = etree.fromstring(xml.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return ""
    if root is None:
        return ""

    parts: list[str] = []

    front = _find_local(root, "front")

    title = _flatten(_find_local(front, "article-title")) if front is not None else ""
    if title:
        parts.append(title)

    authors = _format_authors(front)
    if authors:
        parts.append(authors)

    # Reuse the body walker for the abstract so its (often labelled) section
    # titles and paragraphs land on their own lines instead of concatenating.
    if front is not None:
        for abstract in _iter_local(front, "abstract"):
            _walk_body(abstract, parts)

    body = _find_local(root, "body")
    if body is not None:
        _walk_body(body, parts)

    # Some publishers collect figures/tables in a <floats-group> (a sibling of
    # <body>, after it in document order) rather than placing them inline, so
    # grab those captions too. The bibliography (<back>) is intentionally
    # skipped.
    floats = _find_local(root, "floats-group")
    if floats is not None:
        _walk_body(floats, parts)

    return "\n".join(parts)
