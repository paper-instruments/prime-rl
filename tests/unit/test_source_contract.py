import importlib.metadata
import json
import shutil
from pathlib import Path

import renderers
import verifiers

AGENTS_SDK_COMMIT = "801cc72e4a80acf5e9d6252fb353f2654076b03b"
AGENTS_SDK_REVISION = "801cc72"


def test_verifiers_and_renderers_import_from_vendored_sources():
    root = Path(__file__).resolve().parents[2]

    assert Path(verifiers.__file__).resolve().is_relative_to(root / "deps" / "verifiers")
    assert Path(renderers.__file__).resolve().is_relative_to(root / "deps" / "renderers")


def test_openai_agents_resolves_to_patched_fork_commit():
    direct_url_text = importlib.metadata.distribution("openai-agents").read_text(
        "direct_url.json"
    )

    assert direct_url_text is not None
    direct_url = json.loads(direct_url_text)
    assert direct_url["url"] == (
        "https://github.com/paper-instruments/openai-agents-python.git"
    )
    assert direct_url["vcs_info"] == {
        "vcs": "git",
        "commit_id": AGENTS_SDK_COMMIT,
        "requested_revision": AGENTS_SDK_REVISION,
    }


def test_default_inference_router_is_installed():
    assert shutil.which("vllm-router") is not None
