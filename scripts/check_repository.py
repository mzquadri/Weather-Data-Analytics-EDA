"""Check that the documented source and reference artifacts are available."""

from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "src/generate_data.py",
    "src/eda_analysis.py",
    "results/key_findings.json",
    "results/city_summary_statistics.csv",
    "results/figures/01_temp_violin.png",
    "results/figures/12_pairplot.png",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {', '.join(missing)}")

    for source in (ROOT / "src").glob("*.py"):
        py_compile.compile(source, doraise=True)

    print(f"Repository check passed: {len(REQUIRED_FILES)} required artifacts available.")


if __name__ == "__main__":
    main()
