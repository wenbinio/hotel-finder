import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


@dataclass(frozen=True)
class StaticVerification:
    missing: tuple[str, ...]
    orphaned: tuple[str, ...]
    untracked: tuple[str, ...]


def referenced_assets(index_text: str) -> set[str]:
    assets = set(re.findall(r'(?:src|href)="/([^"?#]+)', index_text))
    for asset in assets:
        path = PurePosixPath(asset)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"static asset path must not escape static/: {asset}")
    return assets


def tracked_static_files(repo_root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "static"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "static/"
    return {
        name.removeprefix(prefix)
        for name in completed.stdout.split("\0")
        if name.startswith(prefix)
    }


def verify_static_tree(static: Path, tracked_files: set[str]) -> StaticVerification:
    index = static / "index.html"
    references = referenced_assets(index.read_text("utf-8"))
    static_root = static.resolve()
    for name in references:
        target = (static / name).resolve()
        try:
            target.relative_to(static_root)
        except ValueError as error:
            raise ValueError(f"static asset path must not escape static/: {name}") from error

    actual_files = {
        path.relative_to(static).as_posix() for path in static.rglob("*") if path.is_file()
    }
    generated_files = {
        name for name in actual_files if PurePosixPath(name).parts[0] == "assets"
    }
    generated_references = {
        name for name in references if PurePosixPath(name).parts[0] == "assets"
    }
    return StaticVerification(
        missing=tuple(sorted(references - actual_files)),
        orphaned=tuple(sorted(generated_files - generated_references)),
        untracked=tuple(sorted(actual_files - tracked_files)),
    )


def main() -> int:
    index = STATIC / "index.html"
    if not index.is_file():
        print("static/index.html is missing", file=sys.stderr)
        return 1
    try:
        verification = verify_static_tree(STATIC, tracked_static_files(ROOT))
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
