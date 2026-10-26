import os
import sys

import pytest

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Skip entire module if websearch dependencies are not available
try:
    from envs.websearch_env.models import WebSearchAction, WebSearchObservation
    from envs.websearch_env.server import WebSearchEnvironment

    WEBSEARCH_AVAILABLE = True
except ImportError:
    WEBSEARCH_AVAILABLE = False
    WebSearchEnvironment = None
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
