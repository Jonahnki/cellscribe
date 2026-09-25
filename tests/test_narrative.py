"""The narrative module must be fully optional and never see raw data."""

import json
import types

import pytest

from cellscribe import narrative
from cellscribe.config import NarrativeConfig
from cellscribe.report.render import render_html


class _FakeMessages:
    def __init__(self, text="Overall the data look good.\n\nCluster 7 is damaged.\n\nFilter and re-cluster.", stop="end_turn", exc=None):
        self.calls = []
        self.text, self.stop, self.exc = text, stop, exc

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return types.SimpleNamespace(
            stop_reason=self.stop,
            model=kwargs["model"],
            content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=self.text)],
        )


class _FakeClient:
    def __init__(self, **kw):
        self.messages = _FakeMessages(**kw)
        self.beta = types.SimpleNamespace(messages=_FakeMessages(**kw))


def test_skipped_without_api_key(sc_result):
    res = narrative.generate(sc_result.summary, NarrativeConfig())
    assert res.status == "skipped" and "ANTHROPIC_API_KEY" in res.reason
    assert res.paragraphs == []


def test_disabled_never_calls_api(sc_result, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = _FakeClient()
    res = narrative.generate(sc_result.summary, NarrativeConfig(enabled=False), client=client)
    assert res.status == "skipped"
    assert client.beta.messages.calls == [] and client.messages.calls == []


def test_report_without_key_has_no_narrative_section(sc_result):
    assert sc_result.summary["narrative"]["status"] == "skipped"
    html = render_html(sc_result)
    assert 'id="narrative"' not in html
    assert 'href="#narrative"' not in html
    assert 'id="methods"' in html


def test_payload_contains_only_summary_statistics(sc_result):
    payload = narrative.build_payload(sc_result.summary)
    text = json.dumps(payload)
    assert len(text) < 30_000
    barcodes = sc_result.adata.obs_names[:1000]
    assert not any(b in text for b in barcodes)
    assert {"sample", "qc", "clustering", "clusters"} <= set(payload)
    for c in payload["clusters"]:
        assert set(c) >= {"cluster", "label", "confidence", "top_marker_genes", "qc_status"}
        assert all(isinstance(g, str) for g in c["top_marker_genes"])


def test_mocked_generation_uses_fallbacks_and_renders(sc_result, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = _FakeClient(text="The data look <b>good</b>.\n\n**Be careful** with cluster 7.")
    res = narrative.generate(sc_result.summary, NarrativeConfig(), client=client)
    assert res.status == "generated" and res.model == "claude-opus-5"
    assert res.paragraphs == ["The data look <b>good</b>.", "Be careful with cluster 7."]
    call = client.beta.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["system"] == narrative.SYSTEM_PROMPT
    user = call["messages"][0]["content"]
    assert json.loads(user.split("\n\n", 1)[1]) == json.loads(json.dumps(narrative.build_payload(sc_result.summary), default=str))

    import copy

    r = copy.copy(sc_result)
    r.summary = dict(sc_result.summary, narrative=res.to_dict())
    r.narrative = res
    html = render_html(r)
    assert 'id="narrative"' in html and "AI narrative summary" in html
    assert "&lt;b&gt;good&lt;/b&gt;" in html  # model output is escaped, never injected as HTML
    assert html.index('id="narrative"') < html.index('id="methods"')


def test_other_models_use_plain_messages(sc_result):
    client = _FakeClient()
    narrative.generate(sc_result.summary, NarrativeConfig(model="claude-sonnet-5"), client=client)
    assert client.messages.calls and "fallbacks" not in client.messages.calls[0]


@pytest.mark.parametrize(
    "kw,expected",
    [({"stop": "refusal"}, "declined"), ({"exc": RuntimeError("network down")}, "failed"), ({"text": "   "}, "no text")],
)
def test_failures_are_graceful(sc_result, kw, expected):
    res = narrative.generate(sc_result.summary, NarrativeConfig(), client=_FakeClient(**kw))
    assert res.status == "failed" and (expected in res.reason or expected == res.status)


def test_api_errors_map_to_messages(sc_result):
    anthropic = pytest.importorskip("anthropic")
    try:
        import httpx2 as httpx  # anthropic >= 1.0
    except ImportError:  # pragma: no cover
        import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
    res = narrative.generate(sc_result.summary, NarrativeConfig(), client=_FakeClient(exc=err))
    assert res.status == "failed" and "API key" in res.reason
