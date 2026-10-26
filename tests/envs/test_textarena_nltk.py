"""NLTK startup compatibility without changing process-wide proxy settings."""

import os
import subprocess
import sys
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textarena_env.server import environment


@pytest.fixture
def nltk_download(monkeypatch, tmp_path):
    nltk = SimpleNamespace(
        __version__="3.10.3",
        download=Mock(),
        data=SimpleNamespace(find=Mock()),
        downloader=SimpleNamespace(
            Downloader=lambda: SimpleNamespace(
                default_download_dir=lambda: str(tmp_path)
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setattr(environment, "_NLTK_DOWNLOADED", False)
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.setattr(
        urllib.request, "getproxies", urllib.request.getproxies_environment
    )
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            monkeypatch.delenv(key)
    return nltk


@pytest.mark.parametrize("proxy_key", ["NO_PROXY", "no_proxy"])
def test_exclusion_only_download_is_isolated(monkeypatch, nltk_download, proxy_key):
    monkeypatch.setenv(proxy_key, "localhost,127.0.0.1")
    original_env = dict(os.environ)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    environment._ensure_nltk_data()
    environment._ensure_nltk_data()

    run.assert_called_once()
    command = run.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "nltk.downloader"]
    assert "--exit-on-error" in command
    assert command[-2:] == ["words", "averaged_perceptron_tagger_eng"]
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["env"] == {
        key: value for key, value in original_env.items() if key != proxy_key
    }
    assert dict(os.environ) == original_env
    nltk_download.download.assert_not_called()


@pytest.mark.parametrize(
    "proxy_key", ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
)
def test_carrying_proxy_keeps_nltk_security_checks(
    monkeypatch, nltk_download, proxy_key
):
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setenv(proxy_key, "http://proxy.example:3128")
    nltk_download.download.side_effect = PermissionError("proxied fetch")
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(PermissionError, match="proxied fetch"):
        environment._ensure_nltk_data()

    run.assert_not_called()
    nltk_download.download.assert_called_once_with(
        "words", quiet=True, raise_on_error=True
    )
    assert not environment._NLTK_DOWNLOADED


def test_explicit_proxy_is_not_bypassed(monkeypatch, nltk_download):
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setattr(
        urllib.request,
        "_opener",
        urllib.request.build_opener(
            urllib.request.ProxyHandler({"https": "http://proxy.example:3128"})
        ),
    )
    nltk_download.download.side_effect = PermissionError("proxied fetch")
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(PermissionError, match="proxied fetch"):
        environment._ensure_nltk_data()
    run.assert_not_called()


def test_failed_subprocess_does_not_cache_success(monkeypatch, nltk_download):
    monkeypatch.setenv("NO_PROXY", "localhost")
    run = Mock(side_effect=subprocess.CalledProcessError(1, "nltk.downloader"))
    monkeypatch.setattr(subprocess, "run", run)
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            environment._ensure_nltk_data()
        assert not environment._NLTK_DOWNLOADED
    assert run.call_count == 2


@pytest.mark.parametrize("version,proxies", [("3.10.3", False), ("3.10.4", True)])
def test_unaffected_download_uses_normal_path(
    monkeypatch, nltk_download, version, proxies
):
    nltk_download.__version__ = version
    if proxies:
        monkeypatch.setenv("NO_PROXY", "localhost")
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    environment._ensure_nltk_data()
    run.assert_not_called()
    assert nltk_download.download.call_count == 2
    assert environment._NLTK_DOWNLOADED


def test_exclusion_only_global_opener_is_not_a_carrying_proxy(
    monkeypatch, nltk_download
):
    monkeypatch.setenv("NO_PROXY", "localhost")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"no": "localhost"})
    )
    monkeypatch.setattr(urllib.request, "_opener", opener)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    environment._ensure_nltk_data()

    run.assert_called_once()
    nltk_download.download.assert_not_called()
    assert urllib.request._opener is opener


@pytest.mark.parametrize(
    "missing_resource", ["corpora/words", "taggers/averaged_perceptron_tagger_eng"]
)
def test_cli_zero_exit_without_corpora_does_not_cache_success(
    monkeypatch, nltk_download, missing_resource
):
    monkeypatch.setenv("NO_PROXY", "localhost")
    # NLTK's CLI can exit zero after reporting a download/security error.
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)

    def find(resource):
        if resource == missing_resource:
            raise LookupError("corpus missing")
        return resource

    nltk_download.data.find.side_effect = find
    for _ in range(2):
        with pytest.raises(LookupError, match="corpus missing"):
            environment._ensure_nltk_data()
        assert not environment._NLTK_DOWNLOADED
    assert run.call_count == 2
