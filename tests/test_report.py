import json
import re
from html.parser import HTMLParser

import pytest

from cellscribe.report.render import citation, methods_paragraphs, render_html, render_report

CORE_SECTIONS = [
    ("summary", "Executive summary"),
    ("input", "Sample &amp; input"),
    ("qc", "Quality control"),
    ("clustering", "Clustering overview"),
    ("cluster-qc", "Per-cluster QC"),
    ("annotation", "Cell-type annotation"),
    ("markers", "Marker genes"),
]
TAIL_SECTIONS = [("methods", "Methods &amp; parameters"), ("cite", "How to cite")]
VOID = {"meta", "br", "img", "input", "link", "hr", "col", "area", "base", "wbr", "source"}


class _Checker(HTMLParser):
    """Collects tags and verifies that non-void elements are properly nested."""

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.external = [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("src", "href"):
            v = attrs.get(key) or ""
            if re.match(r"^(https?:)?//", v) and tag in ("script", "link", "img"):
                self.external.append(v)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def _check(html):
    p = _Checker()
    p.feed(html)
    return p


@pytest.fixture(scope="module")
def sc_html(sc_result):
    return render_html(sc_result)


def test_report_is_valid_self_contained_html(sc_html):
    assert sc_html.startswith("<!DOCTYPE html>")
    p = _check(sc_html)
    assert p.errors == [] and p.stack == []
    assert p.external == []  # no external scripts/styles: opens offline
    assert "Plotly.newPlot" in sc_html and len(sc_html) > 1_000_000  # plotly.js embedded inline


def test_sections_present_in_order(sc_html):
    positions = []
    for sid, title in CORE_SECTIONS + TAIL_SECTIONS:
        m = re.search(rf'<section class="card[^"]*" id="{sid}">\s*<h2><span class="num">\d+</span>{re.escape(title)}</h2>', sc_html)
        assert m, f"section {sid} missing"
        positions.append(m.start())
    assert positions == sorted(positions)
    assert 'id="spatial"' not in sc_html  # no coordinates -> no spatial section


def test_report_content(sc_result, sc_html):
    assert "Research use only" in sc_html and "not validated for clinical" in sc_html
    for c in sc_result.annotation.clusters:
        assert c.label in sc_html
    assert "likely artefact" in sc_html  # damaged-cell cluster flagged
    assert sc_html.count("Plotly.newPlot(") >= 7
    for key in ("qc.nmads", "clustering.resolution_grid", "narrative.model"):
        assert key in sc_html  # every parameter listed for reproducibility
    assert "@software{cellscribe" in sc_html


def test_spatial_report_sections(spatial_result, visium_result):
    for res in (spatial_result, visium_result):
        html = render_html(res)
        p = _check(html)
        assert p.errors == [] and p.external == []
        assert re.search(r'id="spatial">\s*<h2><span class="num">8</span>Spatial analysis', html)
        assert html.index('id="markers"') < html.index('id="spatial"') < html.index('id="methods"')
    vis = render_html(visium_result)
    # H&E image embedded as a data URI (plotly JSON-escapes "/" as \u002f)
    assert "data:image/jpeg;base64," in vis or "data:image\\u002fjpeg;base64," in vis
    assert "Genes per spot" in vis


def test_cdn_mode_is_small(sc_result):
    import copy

    r = copy.copy(sc_result)
    r.config = sc_result.config.with_overrides({"report.plotlyjs": "cdn"})
    html = render_html(r)
    assert "cdn.plot.ly" in html and len(html) < 2_500_000


def test_render_report_writes_file(sc_result, tmp_path):
    out = render_report(sc_result, tmp_path / "sub" / "r.html")
    assert out.exists() and out.stat().st_size > 1_000_000


def test_methods_and_citation(sc_result):
    paras, refs = methods_paragraphs(sc_result.summary)
    text = " ".join(paras)
    assert "median absolute deviation" in text and "Leiden" in text and "Scrublet" in text
    assert any("Wolf" in r for r in refs) and any("Traag" in r for r in refs)
    cite_text, bib = citation()
    assert "Cellscribe" in cite_text and bib.startswith("@software{cellscribe")


def test_summary_is_json_serialisable(sc_result):
    s = json.dumps(sc_result.summary)
    assert "key_findings" in json.loads(s)
