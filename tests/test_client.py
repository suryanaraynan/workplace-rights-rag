from workplace_rag.ingest.govuk_client import GovUKClient


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = responses, [], {}

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        return self.responses[url]


def test_content_is_cached_after_first_fetch(tmp_path):
    url = "https://x/api/content/page"
    s = FakeSession({url: FakeResp(200, {"title": "T"})})
    c = GovUKClient("https://x", tmp_path, delay=0, session=s)
    assert c.get_content("/page") == {"title": "T"}
    assert c.get_content("/page") == {"title": "T"}
    assert len(s.calls) == 1  # second call served from disk


def test_404_returns_none(tmp_path):
    url = "https://x/api/content/missing"
    c = GovUKClient("https://x", tmp_path, delay=0, session=FakeSession({url: FakeResp(404, {})}))
    assert c.get_content("/missing") is None
