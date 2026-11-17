import json
from pathlib import Path

from workplace_rag.ingest.parse import html_to_sections, parse_content

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_guide_parts_get_their_own_deep_links():
    doc = parse_content(load("guide_sick_pay.json"))
    urls = {s.url for s in doc.sections}
    assert "https://www.gov.uk/statutory-sick-pay/overview" in urls
    assert "https://www.gov.uk/statutory-sick-pay/what-youll-get" in urls


def test_html_variant_is_preferred_over_govspeak():
    doc = parse_content(load("guide_sick_pay.json"))
    assert all("ignored" not in s.text for s in doc.sections)


def test_tables_keep_numbers_and_headings_split_sections():
    doc = parse_content(load("guide_sick_pay.json"))
    by_heading = {s.heading: s.text for s in doc.sections}
    assert "£118.75" in by_heading["Rates"]
    assert by_heading["How long"] == "Up to 28 weeks."


def test_answer_format_and_metadata():
    doc = parse_content(load("answer_rest_breaks.json"))
    assert doc.doc_id == "rest-breaks-work"
    assert doc.topics == []  # missing taxons must not crash
    assert doc.sections[0].heading == "Rest breaks at work"  # intro uses page title
    assert "20 minute" in doc.sections[1].text


def test_topics_are_extracted_and_sorted():
    doc = parse_content(load("guide_sick_pay.json"))
    assert doc.topics == ["Employing people", "Sick pay"]


def test_nested_list_text_not_duplicated():
    html = "<ul><li><p>Outer point</p><ul><li>Inner</li></ul></li></ul>"
    text = html_to_sections(html, "T", "u")[0].text
    assert text.count("Inner") == 1 and text.count("Outer point") == 1
