from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "bootstrap-policy-corpus.py"
    spec = importlib.util.spec_from_file_location("bootstrap_policy_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bootstrap_corpus_has_route_status_diversity_and_clean_repositories(tmp_path: Path) -> None:
    module = _module()
    fixtures = module._fixtures()
    assert len(fixtures) == 16
    assert {fixture.expected_route for fixture in fixtures} == {"fast", "deep", "repair"}
    assert {fixture.expected_terminal for fixture in fixtures} == {"completed", "failed"}

    root = tmp_path / "corpus"
    for fixture in fixtures:
        repo = root / fixture.name
        module._write_repo(repo, fixture, reset=False)
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert status == ""
        marker = json.loads((repo / ".git" / "graph-bootstrap-fixture.json").read_text())
        assert marker["expected_route"] == fixture.expected_route
        assert marker["expected_terminal"] == fixture.expected_terminal
