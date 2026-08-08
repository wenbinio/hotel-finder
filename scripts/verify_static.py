import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


def referenced_assets(index_text: str) -> set[str]:
    assets = set(re.findall(r'(?:src|href)="/([^"?#]+)', index_text))
    for asset in assets:
        path = PurePosixPath(asset)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"static asset path must not escape static/: {asset}")
    return assets


def main() -> int:
    index = STATIC / "index.html"
    if not index.is_file():
        print("static/index.html is missing", file=sys.stderr)
        return 1
    try:
        assets = referenced_assets(index.read_text("utf-8"))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    static_root = STATIC.resolve()
    missing = []
    for name in assets:
        target = (STATIC / name).resolve()
        try:
            target.relative_to(static_root)
        except ValueError:
            print(f"static asset path must not escape static/: {name}", file=sys.stderr)
            return 1
        if not target.is_file():
            missing.append(name)
    if missing:
        print("missing static assets: " + ", ".join(sorted(missing)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
