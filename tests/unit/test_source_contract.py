import importlib.metadata
import json
from pathlib import Path

import renderers
import verifiers

AGENTS_SDK_COMMIT = "cbbdca5817f4213780f03bf965b888fdd0a3124a"


def test_verifiers_and_renderers_import_from_vendored_sources():
    root = Path(__file__).resolve().parents[2]

    assert Path(verifiers.__file__).resolve().is_relative_to(root / "deps" / "verifiers")
    assert Path(renderers.__file__).resolve().is_relative_to(root / "deps" / "renderers")


def test_openai_agents_resolves_to_patched_fork_commit():
    direct_url = json.loads(importlib.metadata.distribution("openai-agents").read_text("direct_url.json"))

    assert direct_url["url"] == ("https://github.com/paper-instruments/openai-agents-python.git")
    assert direct_url["vcs_info"] == {
        "vcs": "git",
        "commit_id": AGENTS_SDK_COMMIT,
        "requested_revision": "cbbdca",
    }
