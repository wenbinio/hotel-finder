import json
import shlex
from pathlib import Path

import pytest

from scripts import verify_static

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_source_and_hostable_build_are_tracked():
    package = json.loads((ROOT / "frontend" / "package.json").read_text("utf-8"))
    assert package["scripts"]["build:hostable"] == "vite build --outDir ../static --emptyOutDir"
    assert (ROOT / "frontend" / "src" / "App.jsx").is_file()
    assert (ROOT / "frontend" / "package-lock.json").is_file()


def test_static_verifier_exists():
    assert (ROOT / "scripts" / "verify_static.py").is_file()


def _option(tokens: list[str], name: str) -> str:
    assert tokens.count(name) == 1
    return tokens[tokens.index(name) + 1]


def _named_workflow_step(workflow: str, name: str) -> dict[str, str]:
    marker = f"      - name: {name}\n"
    assert workflow.count(marker) == 1
    block = workflow.split(marker, 1)[1].split("\n      - ", 1)[0]
    values: dict[str, str] = {}
    lines = block.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        prefix = "        "
        if not line.startswith(prefix) or ":" not in line:
            index += 1
            continue
        key, value = line[len(prefix) :].split(":", 1)
        value = value.strip()
        if key == "run" and value == "|":
            commands = []
            index += 1
            while index < len(lines) and lines[index].startswith("          "):
                commands.append(lines[index][10:])
                index += 1
            values[key] = "\n".join(commands)
            continue
        values[key] = value
        index += 1
    return values


def test_deployment_uses_exact_threaded_gunicorn_commands():
    dockerfile = (ROOT / "Dockerfile").read_text("utf-8")
    render = (ROOT / "render.yaml").read_text("utf-8")

    docker_command = next(line for line in dockerfile.splitlines() if line.startswith("CMD "))
    docker_tokens = json.loads(docker_command.removeprefix("CMD "))
    render_command = next(
        line.strip().removeprefix("startCommand: ")
        for line in render.splitlines()
        if line.strip().startswith("startCommand: ")
    )
    render_tokens = shlex.split(render_command)

    for tokens, bind in (
        (docker_tokens, "0.0.0.0:5001"),
        (render_tokens, "0.0.0.0:$PORT"),
    ):
        assert tokens[0] == "gunicorn"
        assert _option(tokens, "--bind") == bind
        assert _option(tokens, "--workers") == "1"
        assert _option(tokens, "--threads") == "8"
        assert _option(tokens, "--timeout") == "120"
        assert _option(tokens, "--access-logfile") == "-"
        assert _option(tokens, "--error-logfile") == "-"
        assert tokens[-1] == "app:app"


def test_deployment_exposes_and_checks_the_application_port():
    dockerfile = (ROOT / "Dockerfile").read_text("utf-8")
    render = (ROOT / "render.yaml").read_text("utf-8")
    assert "EXPOSE 5001" in dockerfile.splitlines()
    healthcheck = next(
        line for line in dockerfile.splitlines() if line.startswith("HEALTHCHECK ")
    )
    assert "http://127.0.0.1:5001/api/health" in healthcheck
    assert "healthCheckPath: /api/health" in {line.strip() for line in render.splitlines()}


def test_ci_runs_every_local_gate():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    assert "    runs-on: ubuntu-latest" in ci.splitlines()
    assert (
        '      - uses: actions/setup-python@v5\n        with:\n          python-version: "3.13"'
        in ci
    )
    assert (
        '      - uses: actions/setup-node@v4\n        with:\n          node-version: "22"'
        in ci
    )

    expected_steps = {
        "Install Python dependencies": {
            "run": "python -m pip install --upgrade pip\n"
            "python -m pip install -r requirements.txt -r requirements-dev.txt",
        },
        "Test Python": {"run": "python -m pytest --cov=app --cov=hotel_finder"},
        "Lint Python": {"run": "python -m ruff check ."},
        "Compile Python": {"run": "python -m compileall -q app.py hotel_finder"},
        "Install frontend dependencies": {
            "working-directory": "frontend",
            "run": "npm ci --no-audit --no-fund",
        },
        "Lint frontend": {"working-directory": "frontend", "run": "npm run lint"},
        "Test frontend": {"working-directory": "frontend", "run": "npm test"},
        "Build hostable frontend": {
            "working-directory": "frontend",
            "run": "npm run build:hostable",
        },
        "Verify static bundle references": {
            "run": "python scripts/verify_static.py --check"
        },
        "Require committed generated assets": {"run": "git diff --exit-code -- static"},
    }
    for name, expected in expected_steps.items():
        step = _named_workflow_step(ci, name)
        assert step == expected


def test_readme_states_remote_exposure_security_boundary():
    readme = (ROOT / "README.md").read_text("utf-8").lower()
    assert "render web service url is publicly reachable" in readme
    assert "cors is not authentication" in readme
    assert "authentication-capable reverse proxy" in readme
    assert "access gateway" in readme
    assert "private network" in readme
    assert "local or vpn-only" in readme
    assert "docker run --rm --publish 127.0.0.1:5001:5001" in readme


def test_readme_documents_defaults_caches_and_job_limits():
    readme = (ROOT / "README.md").read_text("utf-8").lower()
    for phrase in (
        "tomorrow and checkout is five nights later",
        "tomorrow through 90 days later",
        "six one-night stays",
        "cached for 10 minutes",
        "cached for 15 minutes",
        "cached for 60 seconds",
        "four upstream destination searches",
        "from 1 through 6",
        "only one date sweep",
        "capped at 200 logical hotel searches",
        "remain for 30 minutes",
        "no more than eight retained records",
        "restarting this service clears them",
    ):
        assert phrase in readme


def test_readme_documents_the_approved_upgrade_sequence():
    readme = (ROOT / "README.md").read_text("utf-8").lower()
    for phrase in (
        "sqlite",
        "playwright",
        "provider contract monitors",
        "redis and multiple workers only when state must be shared",
        "before any remote exposure",
        "provider apis, locales, currency handling",
    ):
        assert phrase in readme


def test_static_verifier_rejects_asset_path_escapes():
    with pytest.raises(ValueError, match="must not escape"):
        verify_static.referenced_assets('<script src="/../outside.js"></script>')


def _write_static_tree(
    tmp_path: Path,
    index: str,
    files: tuple[str, ...],
    manifest: dict[str, object] | None = None,
) -> Path:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(index, encoding="utf-8")
    for name in files:
        target = static / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name, encoding="utf-8")
    if manifest is not None:
        manifest_path = static / ".vite" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return static


GRAPH_MANIFEST = {
    "index.html": {
        "file": "assets/entry.js",
        "isEntry": True,
        "src": "index.html",
        "imports": ["_vendor.js"],
        "dynamicImports": ["src/lazy.jsx"],
        "css": ["assets/entry.css"],
        "assets": ["assets/bundled-logo.svg"],
    },
    "_vendor.js": {
        "file": "assets/vendor.js",
        "css": ["assets/vendor.css"],
        "assets": ["assets/font.woff2"],
    },
    "src/lazy.jsx": {
        "file": "assets/lazy.js",
        "isDynamicEntry": True,
        "css": ["assets/lazy.css"],
        "assets": ["assets/lazy.png"],
    },
}
GRAPH_FILES = (
    "favicon.svg",
    "icons.svg",
    "assets/entry.js",
    "assets/entry.css",
    "assets/bundled-logo.svg",
    "assets/vendor.js",
    "assets/vendor.css",
    "assets/font.woff2",
    "assets/lazy.js",
    "assets/lazy.css",
    "assets/lazy.png",
)
GRAPH_INDEX = (
    '<link rel="icon" href="/favicon.svg">'
    '<script src="/assets/entry.js"></script>'
    '<link rel="stylesheet" href="/assets/entry.css">'
)
PUBLIC_FILES = {"favicon.svg", "icons.svg"}


def _graph_fixture(
    tmp_path: Path,
    *,
    missing: str | None = None,
    extra: tuple[str, ...] = (),
    manifest: dict[str, object] | None = None,
) -> tuple[Path, set[str]]:
    files = tuple(name for name in GRAPH_FILES if name != missing) + extra
    static = _write_static_tree(
        tmp_path,
        GRAPH_INDEX,
        files,
        GRAPH_MANIFEST if manifest is None else manifest,
    )
    tracked = {"index.html", ".vite/manifest.json", *files}
    return static, tracked


def test_static_verifier_accepts_complete_split_manifest_graph(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path)
    result = verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)
    assert result == verify_static.StaticVerification((), (), ())


def test_static_verifier_rejects_stale_html_entry_asset(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path, extra=("assets/old.js",))
    (static / "index.html").write_text(
        '<link rel="icon" href="/favicon.svg">'
        '<script src="/assets/old.js"></script>'
        '<link rel="stylesheet" href="/assets/entry.css">',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index.html.*assets/entry.js"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


def test_static_verifier_requires_entry_css_in_html(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path)
    (static / "index.html").write_text(
        '<link rel="icon" href="/favicon.svg">'
        '<script src="/assets/entry.js"></script>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index.html.*assets/entry.css"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


def test_static_verifier_rejects_html_asset_outside_manifest_graph(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path, extra=("assets/old.js",))
    (static / "index.html").write_text(
        GRAPH_INDEX + '<script src="/assets/old.js"></script>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside.*assets/old.js"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


def test_static_verifier_accepts_nested_public_asset_reference(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path, extra=("assets/logo.svg",))
    (static / "index.html").write_text(
        GRAPH_INDEX + '<img src="/assets/logo.svg">',
        encoding="utf-8",
    )
    public_files = {*PUBLIC_FILES, "assets/logo.svg"}
    result = verify_static.verify_static_tree(static, tracked, public_files)
    assert result == verify_static.StaticVerification((), (), ())


def test_static_verifier_rejects_tracked_root_reference_outside_whitelist(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path, extra=("old.js",))
    (static / "index.html").write_text(
        GRAPH_INDEX + '<script src="/old.js"></script>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside.*old.js"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


@pytest.mark.parametrize(
    "missing",
    ("assets/vendor.js", "assets/font.woff2", "assets/lazy.png"),
)
def test_static_verifier_reports_missing_transitive_outputs(tmp_path: Path, missing: str):
    static, tracked = _graph_fixture(tmp_path, missing=missing)
    result = verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)
    assert result.missing == (missing,)


def test_static_verifier_rejects_unknown_imported_manifest_entry(tmp_path: Path):
    manifest = json.loads(json.dumps(GRAPH_MANIFEST))
    manifest["index.html"]["imports"] = ["_missing.js"]
    static, tracked = _graph_fixture(tmp_path, manifest=manifest)
    with pytest.raises(ValueError, match="unknown manifest entry"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


def test_static_verifier_rejects_manifest_path_traversal(tmp_path: Path):
    manifest = json.loads(json.dumps(GRAPH_MANIFEST))
    manifest["_vendor.js"]["assets"] = [r"..\outside.woff2"]
    static, tracked = _graph_fixture(tmp_path, manifest=manifest)
    with pytest.raises(ValueError, match="must not escape"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


def test_static_verifier_requires_vite_manifest(tmp_path: Path):
    static = _write_static_tree(tmp_path, GRAPH_INDEX, GRAPH_FILES)
    tracked = {"index.html", *GRAPH_FILES}
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


def test_static_verifier_rejects_unknown_root_html_reference(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path)
    index = static / "index.html"
    index.write_text(GRAPH_INDEX + '<img src="/missing-root.svg">', encoding="utf-8")
    with pytest.raises(ValueError, match="outside.*missing-root.svg"):
        verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)


def test_static_verifier_reports_orphan_generated_assets_but_not_root_files(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path, extra=("assets/old.js",))
    result = verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)
    assert result.orphaned == ("assets/old.js",)


def test_static_verifier_reports_untracked_assets(tmp_path: Path):
    static, tracked = _graph_fixture(tmp_path)
    tracked.remove("assets/lazy.js")
    result = verify_static.verify_static_tree(static, tracked, PUBLIC_FILES)
    assert result.untracked == ("assets/lazy.js",)
