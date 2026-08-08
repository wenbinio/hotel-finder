import json
from pathlib import Path

import pytest

from scripts.verify_static import referenced_assets

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_source_and_hostable_build_are_tracked():
    package = json.loads((ROOT / "frontend" / "package.json").read_text("utf-8"))
    assert package["scripts"]["build:hostable"] == "vite build --outDir ../static --emptyOutDir"
    assert (ROOT / "frontend" / "src" / "App.jsx").is_file()
    assert (ROOT / "frontend" / "package-lock.json").is_file()


def test_static_verifier_exists():
    assert (ROOT / "scripts" / "verify_static.py").is_file()


def test_deployment_uses_one_threaded_worker():
    dockerfile = (ROOT / "Dockerfile").read_text("utf-8")
    render = (ROOT / "render.yaml").read_text("utf-8")
    for text in (dockerfile, render):
        assert "--workers" in text and "1" in text
        assert "--threads" in text and "8" in text


def test_ci_runs_every_local_gate():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    for command in (
        "pytest",
        "ruff check",
        "npm ci",
        "npm run lint",
        "npm test",
        "npm run build:hostable",
        "verify_static.py",
    ):
        assert command in ci


def test_operator_readme_exists():
    assert (ROOT / "README.md").is_file()


def test_static_verifier_rejects_asset_path_escapes():
    with pytest.raises(ValueError, match="must not escape"):
        referenced_assets('<script src="/../outside.js"></script>')
