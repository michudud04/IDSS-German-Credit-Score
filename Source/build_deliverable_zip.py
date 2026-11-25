"""Build the final PW3 deliverable ZIP archive.

Usage:
    python -m Source.build_deliverable_zip --year 2026-2027 --team 1 --title CreditScoringIDSS
"""

import argparse
from pathlib import Path
import zipfile

from Source import config

SKIP_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store", "cache"}


def _add_directory(zf: zipfile.ZipFile, src_dir: Path, archive_prefix: str) -> int:
    count = 0
    for item in sorted(src_dir.rglob("*")):
        if item.is_dir() or item.suffix == ".pyc":
            continue
        if any(part in SKIP_NAMES for part in item.relative_to(src_dir).parts):
            continue
        zf.write(item, f"{archive_prefix}/{item.relative_to(src_dir).as_posix()}")
        count += 1
    return count


def build_zip(output_path: Path) -> dict[str, int]:
    root = config.ROOT
    counts = {}

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        counts["Documentation"] = _add_directory(zf, root / "Documentation", "Documentation")
        counts["Data"] = _add_directory(zf, config.DATA_DIR, "Data")
        counts["Models"] = _add_directory(zf, config.MODELS_DIR, "Models")
        counts["Source"] = _add_directory(zf, root / "Source", "Source")
        counts["Source"] += _add_directory(zf, root / "notebooks", "Source/notebooks")
        zf.write(root / "requirements.txt", "Source/requirements.txt")
        counts["Source"] += 1
        counts["Demo"] = _add_directory(zf, root / "Demo", "Demo")
        counts["Presentation"] = _add_directory(zf, root / "Presentation", "Presentation")
        zf.write(root / "README.txt", "README.txt")
        counts["README.txt"] = 1

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026-2027")
    parser.add_argument("--team", default="1")
    parser.add_argument("--title", default="CreditScoringIDSS")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    filename = f"PW3-IDSS-{args.year}-T{args.team}-{args.title}.zip"
    out_path = Path(args.output) if args.output else config.ROOT / filename

    counts = build_zip(out_path)
    print(f"Created {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")
    for section, n in counts.items():
        print(f"  {section:15s}: {n} file(s)")
