import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


def referenced_assets(index_text: str) -> set[str]:
    return set(re.findall(r'(?:src|href)="/([^"?#]+)', index_text))


def main() -> int:
    index = STATIC / "index.html"
    if not index.is_file():
        print("static/index.html is missing", file=sys.stderr)
        return 1
    missing = [
        name for name in referenced_assets(index.read_text("utf-8")) if not (STATIC / name).is_file()
    ]
    if missing:
        print("missing static assets: " + ", ".join(sorted(missing)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
