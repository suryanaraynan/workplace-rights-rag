import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ingest", Path(__file__).resolve().parents[1] / "scripts" / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


class FakeClient:
    def __init__(self, pages):
        self.pages = pages

    def get_content(self, path, use_cache=True):
        return self.pages.get(path)


def test_redirects_are_followed():
    c = FakeClient({
        "/old": {"schema_name": "redirect", "redirects": [{"destination": "/new"}]},
        "/new": {"schema_name": "guide", "title": "New"},
    })
    assert ingest.fetch_following_redirects(c, "/old", True)["title"] == "New"


def test_redirect_loops_give_up():
    c = FakeClient({"/a": {"schema_name": "redirect", "redirects": [{"destination": "/a"}]}})
    assert ingest.fetch_following_redirects(c, "/a", True) is None
