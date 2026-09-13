"""Fail if the README stops agreeing with results/summary.json.

    python scripts/check_repository.py

Checks that the documented files exist and compile, that the summary is
internally consistent, and that every number the README quotes still matches the
recorded artifact.

Each claim is matched with its surrounding words included, so a value that has
drifted into a different sentence does not accidentally satisfy a check.
"""

from __future__ import annotations

import calendar
import json
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary.json"

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "src/generate_data.py",
    "src/eda_analysis.py",
    "src/summary.py",
    "tests/test_data.py",
    "tests/test_analysis.py",
    "scripts/figures/generate_figures.py",
    "results/summary.json",
    "results/key_findings.json",
    "docs/figures/01_annual_cycle.png",
    "docs/figures/02_precipitation_by_month.png",
    "docs/figures/03_wind_is_noise.png",
    "docs/figures/04_correlation_pooling.png",
)


def renderer_of(png: Path) -> str:
    """The matplotlib version recorded inside a PNG, or "unknown".

    matplotlib writes its version into a tEXt chunk keyed "Software". Read straight
    from the bytes so this needs no image library.
    """
    data = png.read_bytes()
    marker = b"Matplotlib version"
    start = data.find(marker)
    if start < 0:
        return "unknown"
    tail = data[start + len(marker):start + len(marker) + 20]
    version = bytearray()
    for byte in tail:
        char = chr(byte)
        if char.isdigit() or char == ".":
            version.append(byte)
        else:
            break
    return version.decode() or "unknown"


def figures_were_rendered_together(root: Path) -> tuple[bool, dict[str, list[str]]]:
    """Every committed figure should carry the same renderer version.

    Sixteen figures are committed and two different scripts produce them:
    src/eda_analysis.py writes the twelve under results/figures, and
    scripts/figures/generate_figures.py writes the four under docs/figures.
    Regenerating one set and not the other leaves the repository showing two
    vintages of the same analysis, and nothing would have said so: CI checked
    four of the sixteen for existence and no more.

    Comparing the recorded versions catches that without failing merely because
    matplotlib moved, which is the point. A whole-set regeneration under a new
    version still agrees with itself.
    """
    by_version: dict[str, list[str]] = {}
    for png in sorted(root.rglob("*.png")):
        if ".git" in png.parts:
            continue
        by_version.setdefault(renderer_of(png), []).append(
            str(png.relative_to(root)).replace("\\", "/"))
    return len(by_version) <= 1, by_version


def claims(data: dict) -> list[tuple[str, str]]:
    """Every quoted number, paired with enough context to anchor it."""
    coverage = data["coverage"]
    wind = data["wind_ranking"]
    precipitation = data["monthly_precipitation"]
    example = precipitation["example_of_the_difference"]
    humidity = data["temperature_humidity"]
    pressure = data["temperature_pressure"]
    extremes = data["extremes"]
    uv = data["uv_index_floor"]["share_of_days_at_or_below_0.05"]
    moved = precipitation["months_that_change_rank"]

    out = [
        (f"| Rows | {coverage['rows']:,} |", "the row count"),
        (f"| Cities | {coverage['cities']} |", "the city count"),
        (f"| Columns | {coverage['columns']} |", "the column count"),
        (f"| Distinct days | {coverage['distinct_days']:,} |", "the day count"),
        (f"| First date | {coverage['first_date']} |", "the first date"),
        (f"| Last date | {coverage['last_date']} |", "the last date"),
        (f"{coverage['last_date']}. 2023 is short by "
         f"{coverage['years_short_of_a_full_calendar_year']['2023']} days",
         "the short-year statement"),
        (f"| Largest gap between consecutive days | "
         f"{coverage['largest_gap_between_consecutive_days']} day |",
         "the largest gap"),
        (f"| Duplicate city and date pairs | "
         f"{coverage['duplicate_city_date_pairs']} |", "the duplicate count"),

        (f"New York leads Delhi by **{wind['difference_ms']} m/s, which is\n"
         f"{wind['difference_in_standard_errors']:.2f} standard errors**",
         "the wind difference"),
        (f"full spread across all six cities is {wind['spread_across_all_cities_ms']} m/s",
         "the wind spread"),

        (f"February is observed on\n**{example['february_observed_days']:,} days** "
         f"against January's\n**{example['january_observed_days']:,}**",
         "the observed-day counts"),
        (f"February totals **{example['february_total_mm']:,.1f} mm** against "
         f"January's **{example['january_total_mm']:,.1f} mm**", "the monthly totals"),
        (f"**{example['february_mm_per_day']:.3f} mm**\nagainst "
         f"**{example['january_mm_per_day']:.3f} mm**", "the per-day rates"),
        (f"February sits **{example['february_total_relative_to_january_pct']} "
         f"percent** relative to January, and per observed day\n"
         f"**{example['february_rate_relative_to_january_pct']} percent** above it",
         "the relative differences"),
        (f"Only **"
         f"{' and '.join(calendar.month_name[m] for m in moved)}**\n"
         f"change position", "which months change rank"),

        (f"line falls **{humidity['pooled_slope']}** percent humidity per degree",
         "the pooled slope"),
        (f"the steepest is **{humidity['steepest_city_slope']}** and the shallowest "
         f"**{humidity['shallowest_city_slope']}**", "the per-city slope range"),

        (f"| Pooled across all six cities | {humidity['pooled_across_cities']} |",
         "the pooled correlation"),
        (f"| Within each city, averaged | {humidity['within_city_mean']} |",
         "the within-city correlation"),
        (f"| Within each city and month, averaged | "
         f"{humidity['within_city_and_month_mean']} |",
         "the within-city-and-month correlation"),
        (f"{pressure['pooled_across_cities']} pooled, "
         f"{pressure['within_city_mean']} within city,\n"
         f"{pressure['within_city_and_month_mean']} within city and month",
         "the pressure correlations"),

        (f"| Hottest day | {extremes['hottest_day']['city']} | "
         f"{extremes['hottest_day']['date']} | "
         f"{extremes['hottest_day']['value']} C |", "the hottest day"),
        (f"| Coldest day | {extremes['coldest_day']['city']} | "
         f"{extremes['coldest_day']['date']} | "
         f"{extremes['coldest_day']['value']} C |", "the coldest day"),
        (f"| Wettest day | {extremes['wettest_day']['city']} | "
         f"{extremes['wettest_day']['date']} | "
         f"{extremes['wettest_day']['value']} mm |", "the wettest day"),
        (f"| Windiest day | {extremes['windiest_day']['city']} | "
         f"{extremes['windiest_day']['date']} | "
         f"{extremes['windiest_day']['value']} m/s |", "the windiest day"),
        (f"| Longest dry run | {extremes['longest_dry_runs'][0]['city']} | | "
         f"{extremes['longest_dry_runs'][0]['longest_dry_run_days']} days |",
         "the longest dry run"),

        (f"between {min(uv.values()) * 100:.1f} and {max(uv.values()) * 100:.1f} "
         f"percent of days", "the UV floor share"),
    ]

    for city, stats in sorted(data["cities"].items()):
        out.append((
            f"| {city} | {stats['latitude']} | {stats['hemisphere']} | "
            f"{stats['mean_temperature_c']:.2f} | "
            f"{stats['temperature_standard_deviation_c']:.2f} | "
            f"{stats['coldest_day_c']} | {stats['hottest_day_c']} | "
            f"{stats['mean_annual_precipitation_mm']} | "
            f"{stats['dry_day_share'] * 100:.1f}% |",
            f"the {city} row"))
    return out


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"  missing required files: {', '.join(missing)}")
    for source in sorted((ROOT / "src").glob("*.py")):
        py_compile.compile(source, doraise=True)
    print(f"  {len(REQUIRED_FILES)} required files present, src/ compiles")

    # Counts as well as vintages: a deleted figure leaves a consistent set behind.
    # CI listed four of the sixteen by name and checked nothing about the other twelve.
    for directory, expected in (("docs/figures", 4), ("results/figures", 12)):
        found = len(list((ROOT / directory).glob("*.png")))
        if found != expected:
            raise SystemExit(f"  {directory} holds {found} figures, expected {expected}")

    consistent, by_version = figures_were_rendered_together(ROOT)
    if not consistent:
        print()
        for version, files in sorted(by_version.items()):
            print(f"  matplotlib {version}: {', '.join(files)}")
        raise SystemExit(
            "\n  the committed figures were not all rendered by the same "
            "matplotlib, "
            "so at least one set is stale. Regenerate both: "
            "`python src/eda_analysis.py` and "
            "`python scripts/figures/generate_figures.py`")
    rendered_by = next(iter(by_version))
    total = sum(len(files) for files in by_version.values())
    print(f"  all {total} committed figures were rendered by matplotlib {rendered_by}")

    if not SUMMARY.exists():
        raise SystemExit("  results/summary.json is missing; run python -m src.summary")
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))

    # The wind ranking must stay indistinguishable from noise. If a future change
    # gives wind a city term, the README's central example stops being true.
    if data["wind_ranking"]["difference_in_standard_errors"] >= 2.0:
        raise SystemExit(
            "  the wind difference is now "
            f"{data['wind_ranking']['difference_in_standard_errors']} standard "
            "errors, so the README's claim that it is noise no longer holds")
    print(f"  the wind lead is still "
          f"{data['wind_ranking']['difference_in_standard_errors']} standard errors")

    # Pooling must still inflate the correlation, or the third finding is gone.
    humidity = data["temperature_humidity"]
    if abs(humidity["pooled_across_cities"]) <= abs(
            humidity["within_city_and_month_mean"]):
        raise SystemExit("  pooling no longer inflates the correlation, so the "
                         "README's third finding no longer holds")
    print("  pooling still inflates the temperature and humidity correlation")

    # The figure's left panel asserts the pooled line is steeper than all six.
    if humidity["pooled_slope"] > min(humidity["per_city_slope"].values()):
        raise SystemExit(
            f"  the pooled slope {humidity['pooled_slope']} is no longer steeper "
            f"than every city, so the left panel of figure 04 no longer holds")
    print(f"  the pooled slope {humidity['pooled_slope']} is still steeper than "
          f"every city, the steepest being {humidity['steepest_city_slope']}")

    def flatten(text: str) -> str:
        """Collapse runs of whitespace so a claim can wrap in the README."""
        return re.sub(r"\s+", " ", text).strip()

    flat = flatten((ROOT / "README.md").read_text(encoding="utf-8"))
    failures = []
    checks = claims(data)
    for expected, description in checks:
        wanted = flatten(expected.replace("\\*", "*"))
        if wanted not in flat:
            failures.append(f"  README does not state {description}: "
                            f"expected {wanted!r}")
    print(f"  {len(checks) - len(failures)} of {len(checks)} recorded numbers "
          f"found in the README")

    if failures:
        print()
        for failure in failures:
            print(failure)
        plural = "claim" if len(failures) == 1 else "claims"
        raise SystemExit(f"\n  {len(failures)} {plural} in the README no longer "
                         f"match results/summary.json")

    print("  README and results/summary.json agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
