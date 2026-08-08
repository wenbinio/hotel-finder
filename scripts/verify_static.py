import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
MANIFEST_PATH = ".vite/manifest.json"


@dataclass(frozen=True)
class StaticVerification:
    missing: tuple[str, ...]
    orphaned: tuple[str, ...]
    untracked: tuple[str, ...]


def referenced_assets(index_text: str) -> set[str]:
    return {
        normalized_output_path(asset)
        for asset in re.findall(r'(?:src|href)="/([^"?#]+)', index_text)
    }


def normalized_output_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("static asset path must be a non-empty string")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or ":" in path.parts[0]
    ):
        raise ValueError(f"static asset path must not escape static/: {value}")
    return path.as_posix()


def declared_manifest_outputs(manifest_text: str) -> set[str]:
    manifest = json.loads(manifest_text)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Vite manifest must be a non-empty object")
    for key, record in manifest.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            raise ValueError("Vite manifest entries must be named objects")

    roots = sorted(key for key, record in manifest.items() if record.get("isEntry") is True)
    if not roots:
        raise ValueError("Vite manifest has no entry chunk")

    outputs: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visited:
            return
        record = manifest.get(key)
        if record is None:
            raise ValueError(f"unknown manifest entry: {key}")
        visited.add(key)

        if "file" not in record:
            raise ValueError(f"Vite manifest entry {key!r} has no file")
        outputs.add(normalized_output_path(record["file"]))

        for field in ("css", "assets"):
            values = record.get(field, [])
            if not isinstance(values, list):
                raise ValueError(f"Vite manifest entry {key!r} has invalid {field}")
            outputs.update(normalized_output_path(value) for value in values)

        for field in ("imports", "dynamicImports"):
            values = record.get(field, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Vite manifest entry {key!r} has invalid {field}")
            for imported_key in values:
                visit(imported_key)

    for root in roots:
        visit(root)

    unreachable = sorted(set(manifest) - visited)
    if unreachable:
        raise ValueError("unreachable Vite manifest entries: " + ", ".join(unreachable))
    return outputs


def tracked_static_files(repo_root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "static"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "static/"
    return {
        normalized_output_path(name.removeprefix(prefix))
        for name in completed.stdout.split("\0")
        if name.startswith(prefix)
    }


def public_source_files(public_root: Path) -> set[str]:
    if not public_root.is_dir():
        return set()
    return {
        normalized_output_path(path.relative_to(public_root).as_posix())
        for path in public_root.rglob("*")
        if path.is_file()
    }


def verify_static_tree(
    static: Path,
    tracked_files: set[str],
    public_files: set[str],
) -> StaticVerification:
    index = static / "index.html"
    references = referenced_assets(index.read_text("utf-8"))
    manifest = static / MANIFEST_PATH
    if not manifest.is_file():
        raise FileNotFoundError(f"Vite manifest is missing: {manifest}")
    declared_outputs = declared_manifest_outputs(manifest.read_text("utf-8"))

    expected_files = {
        "index.html",
        MANIFEST_PATH,
        *references,
        *declared_outputs,
        *(normalized_output_path(name) for name in public_files),
    }
    static_root = static.resolve()
    for name in expected_files:
        target = (static / name).resolve()
        try:
            target.relative_to(static_root)
        except ValueError as error:
            raise ValueError(f"static asset path must not escape static/: {name}") from error

    actual_files = {
        path.relative_to(static).as_posix() for path in static.rglob("*") if path.is_file()
    }
    return StaticVerification(
        missing=tuple(sorted(expected_files - actual_files)),
        orphaned=tuple(sorted(actual_files - expected_files)),
        untracked=tuple(sorted(actual_files - tracked_files)),
    )


def main() -> int:
    index = STATIC / "index.html"
    if not index.is_file():
        print("static/index.html is missing", file=sys.stderr)
        return 1
    try:
        verification = verify_static_tree(
            STATIC,
            tracked_static_files(ROOT),
            public_source_files(ROOT / "frontend" / "public"),
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    problems = (
        ("missing static assets", verification.missing),
        ("orphaned generated assets", verification.orphaned),
        ("untracked static assets", verification.untracked),
    )
    failed = False
    for label, names in problems:
        if names:
            print(f"{label}: " + ", ".join(names), file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
