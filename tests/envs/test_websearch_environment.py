import os
import sys

import pytest

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# The server modules import `models` as a top-level module
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../envs/websearch_env")
    ),
)

# Skip entire module if websearch dependencies are not available
try:
    from envs.websearch_env.models import WebSearchAction, WebSearchObservation
    from envs.websearch_env.server import WebSearchEnvironment
    from envs.websearch_env.server.web_search_tool import WebSearchTool

    WEBSEARCH_AVAILABLE = True
except ImportError:
    WEBSEARCH_AVAILABLE = False
    WebSearchEnvironment = None
    WebSearchTool = None
    WebSearchAction = None
    WebSearchObservation = None

pytestmark = pytest.mark.skipif(
    not WEBSEARCH_AVAILABLE,
    reason="websearch_env dependencies not installed (chardet, etc.)",
)


@pytest.mark.skipif(
    not os.environ.get("SERPER_API_KEY"), reason="SERPER_API_KEY not set"
)
def test_websearch_environment():
    # Create the environment
    env = WebSearchEnvironment()

    # Reset the environment
    obs: WebSearchObservation = env.reset()
    assert obs.web_contents == []
    assert obs.content == ""

    # Step the environment
    obs: WebSearchObservation = env.step(
        WebSearchAction(query="What is the capital of France?")
    )
    if not obs.metadata.get("error"):
        assert obs.web_contents != []
        assert len(obs.web_contents) == 5
        assert obs.metadata == {"query": "What is the capital of France?"}
    else:
        assert obs.web_contents == []
        assert "[ERROR]" in obs.content


class _Response:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_deep_search_keeps_snippet_when_page_lacks_it(monkeypatch):
    snippet = "The capital of France is Paris and it is large ... another part of the snippet here is long enough"
    serper = {
        "organic": [{"title": "France", "snippet": snippet, "link": "http://x/1"}]
    }
    page = _Response(content=b"<html>js app shell, nothing useful</html>")
    monkeypatch.setattr("requests.post", lambda *a, **kw: _Response(serper))
    monkeypatch.setattr("requests.get", lambda *a, **kw: page)

    tool = WebSearchTool(api_key="k", snippet_only=False)
    obs = tool.execute(WebSearchAction(query="capital"))

    assert obs.web_contents[0].content == snippet
