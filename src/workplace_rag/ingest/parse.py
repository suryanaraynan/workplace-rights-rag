"""Turn a raw GOV.UK Content API response into a clean, structured Document.

Key design choice: we keep the page's *heading structure* (sections), not
just a flat blob of text. Headings tell us what a paragraph is about, which
later improves chunking, retrieval and citations ("see: Sick pay > Eligibility").
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup

HEADING_TAGS = {"h2", "h3", "h4"}
BLOCK_TAGS = {"p", "li", "tr"}


@dataclass
class Section:
    heading: str
    text: str
    url: str          # deep link: guides have one URL per part
    parent: str = ""  # enclosing h2 when this section starts at an h3/h4

    @property
    def heading_path(self) -> str:
        return f"{self.parent} > {self.heading}" if self.parent else self.heading


@dataclass
class Document:
    doc_id: str
    url: str
    title: str
    description: str
    doc_format: str
    updated_at: str
    topic: str = ""                                   # our curated label
    topics: list[str] = field(default_factory=list)   # GOV.UK's own tags (inconsistent)
    sections: list[Section] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    """Collapse whitespace and undo spacing artifacts from joining inline HTML
    elements, e.g. "( SSP )" -> "(SSP)", "workers ." -> "workers."."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?)\]’])", r"\1", text)
    text = re.sub(r"([(\[‘])\s+", r"\1", text)
    return text.strip()


# Site boilerplate that carries no guidance, seen in real data.
_BOILERPLATE = re.compile(r"This guide is also available in Welsh \(Cymraeg\)\.?\s*", re.IGNORECASE)


# Placeholder sections seen in real data, e.g. "Northern Ireland: [TBC]".
_PLACEHOLDER = re.compile(r"^\[?\s*(tbc|to be confirmed|n/?a)\s*\]?\.?$", re.IGNORECASE)


def is_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER.match(text.strip()))


def _body_to_html(body) -> str:
    """Content API bodies are either an HTML string or a list of
    {content_type, content} variants. Prefer the HTML variant."""
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        for variant in body:
            if variant.get("content_type") == "text/html":
                return variant.get("content", "")
        return body[0].get("content", "") if body else ""
    return ""


def html_to_sections(html: str, default_heading: str, url: str) -> list[Section]:
    """Split HTML into sections at each h2/h3/h4. Text blocks under a heading
    belong to it. Table rows become 'cell | cell' lines, so numbers survive."""
    soup = BeautifulSoup(html, "html.parser")
    sections: list[Section] = []
    # h2 starts as the part/page title, so an h3 before any h2 still has context.
    heading, parent, h2, lines = default_heading, "", default_heading, []

    def flush():
        text = _BOILERPLATE.sub("", _clean(" ".join(lines))).strip()
        if text and not is_placeholder(text):
            sections.append(Section(heading=heading, text=text, url=url, parent=parent))

    for el in soup.find_all(list(HEADING_TAGS | BLOCK_TAGS)):
        if el.name in HEADING_TAGS:
            flush()
            heading, lines = _clean(el.get_text(" ")), []
            if el.name == "h2":
                h2, parent = heading, ""
            else:
                parent = h2  # h3/h4 remembers the h2 it sits under
        elif el.name == "tr":
            cells = [_clean(c.get_text(" ")) for c in el.find_all(["th", "td"])]
            lines.append(" | ".join(c for c in cells if c) + ".")
        elif el.name == "li":
            # Take only this item's own text; nested lists are visited separately.
            own = [c.get_text(" ") if hasattr(c, "get_text") else str(c)
                   for c in el.children if getattr(c, "name", None) not in ("ul", "ol")]
            lines.append(" ".join(own))
        elif el.name == "p" and el.find_parent("li"):
            continue  # already included via the parent <li>
        else:
            lines.append(el.get_text(" "))
    flush()
    return sections


def parse_content(raw: dict, base_url: str = "https://www.gov.uk", topic: str = "") -> Document:
    base_path = raw.get("base_path", "")
    url = base_url + base_path
    title = _clean(raw.get("title", ""))
    details = raw.get("details", {}) or {}

    sections: list[Section] = []
    if details.get("parts"):  # multi-page "guide" format
        for part in details["parts"]:
            part_url = f"{url}/{part['slug']}"
            sections += html_to_sections(_body_to_html(part.get("body")),
                                         default_heading=_clean(part.get("title", title)),
                                         url=part_url)
    else:
        sections = html_to_sections(_body_to_html(details.get("body")),
                                    default_heading=title, url=url)

    taxons = (raw.get("links", {}) or {}).get("taxons", []) or []
    return Document(
        doc_id=base_path.strip("/").replace("/", "__") or raw.get("content_id", ""),
        url=url,
        title=title,
        description=_clean(raw.get("description") or ""),
        doc_format=raw.get("document_type") or raw.get("schema_name", ""),
        updated_at=raw.get("public_updated_at", ""),
        topic=topic,
        topics=sorted({t["title"] for t in taxons if t.get("title")}),
        sections=sections,
    )
