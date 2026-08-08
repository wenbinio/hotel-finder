import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_source_and_hostable_build_are_tracked():
    package = json.loads((ROOT / "frontend" / "package.json").read_text("utf-8"))
    assert package["scripts"]["build:hostable"] == "vite build --outDir ../static --emptyOutDir"
    assert (ROOT / "frontend" / "src" / "App.jsx").is_file()
    assert (ROOT / "frontend" / "package-lock.json").is_file()


def test_static_verifier_exists():
    assert (ROOT / "scripts" / "verify_static.py").is_file()
