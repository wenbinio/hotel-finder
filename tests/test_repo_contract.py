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


def _write_static_tree(tmp_path: Path, index: str, files: tuple[str, ...]) -> Path:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(index, encoding="utf-8")
    for name in files:
        target = static / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name, encoding="utf-8")
    return static


def test_static_verifier_reports_dangling_references(tmp_path: Path):
    static = _write_static_tree(
        tmp_path,
        '<script src="/assets/missing.js"></script>',
        (),
    )
    result = verify_static.verify_static_tree(static, {"index.html"})
    assert result.missing == ("assets/missing.js",)


def test_static_verifier_reports_orphan_generated_assets_but_not_root_files(tmp_path: Path):
    static = _write_static_tree(
        tmp_path,
        '<link rel="icon" href="/favicon.svg"><script src="/assets/app.js"></script>',
        ("favicon.svg", "icons.svg", "assets/app.js", "assets/old.js"),
    )
    tracked = {"index.html", "favicon.svg", "icons.svg", "assets/app.js", "assets/old.js"}
    result = verify_static.verify_static_tree(static, tracked)
    assert result.orphaned == ("assets/old.js",)


def test_static_verifier_reports_untracked_assets(tmp_path: Path):
    static = _write_static_tree(
        tmp_path,
        '<script src="/assets/app.js"></script>',
        ("assets/app.js",),
    )
    result = verify_static.verify_static_tree(static, {"index.html"})
    assert result.untracked == ("assets/app.js",)
