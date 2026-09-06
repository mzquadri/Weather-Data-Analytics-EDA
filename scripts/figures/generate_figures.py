"""Render the README figures from results/summary.json and the generated data.

    python scripts/figures/generate_figures.py

Four figures, each answering one question:

  01  what the annual cycle looks like in each city, including the two hemispheres
  02  which months are wet, and why the answer depends on how you aggregate
  03  whether the windiest city is windier than the next or just luckier
  04  how much of the temperature and humidity relationship is a pooling artifact

The repository already had twelve plots. These replace the ones that carried a
claim, and the originals are kept where they are still descriptive.
"""

from __future__ import annotations

import calendar
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import portfolio_style as ps

SUMMARY = ROOT / "results" / "summary.json"
DATA = ROOT / "data" / "weather_data.csv"
FIGURES = ROOT / "docs" / "figures"

MONTHS = [calendar.month_abbr[m] for m in range(1, 13)]
CITY_COLOUR = {
    "Delhi": ps.RED,
    "Tokyo": ps.BLUE,
    "New York": ps.TEAL,
    "Munich": ps.SLATE,
    "Sydney": ps.AMBER,
    "Nairobi": ps.GREEN,
}


def load():
    if not SUMMARY.exists():
        raise SystemExit("run `python -m src.summary` first")
    if not DATA.is_file():
        raise SystemExit("run `python -m src.generate_data` first")
    return (json.loads(SUMMARY.read_text(encoding="utf-8")),
            pd.read_csv(DATA, parse_dates=["date"]))


def figure_01_annual_cycle(summary: dict, frame: pd.DataFrame) -> None:
    """What does the year look like in each city?"""
    coverage = summary["coverage"]
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    fig.subplots_adjust(left=0.075, right=0.80, top=0.745, bottom=0.235)

    monthly = frame.groupby(["city", "month"])["temp_mean_c"].mean().unstack()
    for city in monthly.index:
        southern = summary["cities"][city]["hemisphere"] == "southern"
        ax.plot(range(1, 13), monthly.loc[city], color=CITY_COLOUR[city],
                linewidth=2.2, marker="o", markersize=4.5,
                linestyle="--" if southern else "-")
        ax.text(12.25, monthly.loc[city].iloc[-1], f" {city}", fontsize=10,
                color=CITY_COLOUR[city], va="center")

    ax.set_xticks(range(1, 13), MONTHS, fontsize=10)
    ax.set_xlim(0.6, 12.4)
    ax.set_ylabel("Mean daily temperature, degrees Celsius", fontsize=11,
                  color=ps.MUTED)
    ax.set_xlabel(f"Month, averaged over {coverage['first_date']} to "
                  f"{coverage['last_date']}", fontsize=10.6, color=ps.MUTED)
    ps.clean(ax, grid_axis="y")

    handles = [
        plt.Line2D([], [], color=ps.FAINT, linewidth=2, linestyle="-",
                   label="northern hemisphere"),
        plt.Line2D([], [], color=ps.FAINT, linewidth=2, linestyle="--",
                   label="southern hemisphere"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10,
              labelcolor=ps.MUTED)

    ps.title_block(
        fig, "Six generated annual cycles, two of them inverted",
        "Monthly mean of the daily mean temperature. Sydney and Nairobi are given "
        "southern latitudes, so their\ncycles run opposite to the other four.")

    nairobi = summary["cities"]["Nairobi"]
    ps.footnote(fig, [
        f"Nairobi sits at latitude {nairobi['latitude']} and is given an annual "
        f"amplitude of 2 degrees, so its line is nearly flat and its season labels "
        f"carry little",
        "meaning. These are generated distributions with city names attached, not "
        "measurements from those cities.",
    ])
    ps.save(fig, FIGURES, "01_annual_cycle")


def figure_02_precipitation(summary: dict, frame: pd.DataFrame) -> None:
    """Which months are wet, and does the answer depend on the aggregation?"""
    precipitation = summary["monthly_precipitation"]
    totals = [precipitation["total_mm_by_month"][str(m)] for m in range(1, 13)]
    rates = [precipitation["mm_per_day_by_month"][str(m)] for m in range(1, 13)]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 6.2))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.70, bottom=0.245, wspace=0.24)

    example = precipitation["example_of_the_difference"]
    positions = np.arange(1, 13)
    highlight = (1, 2)

    def panel(ax, values, base_colour, label, heading, fmt):
        ax.bar(positions, values, width=0.62,
               color=[ps.AMBER if m in highlight else base_colour
                      for m in positions])
        # Only January and February carry a value. The whole point is that the
        # difference between those two bars is not readable by eye.
        headroom = max(values) * 0.035
        for month in highlight:
            ax.text(month, values[month - 1] + headroom, fmt.format(values[month - 1]),
                    ha="center", fontsize=10.2, color=ps.INK, fontweight="600")
        ax.set_ylim(0, max(values) * 1.16)
        ax.set_xticks(positions, MONTHS, fontsize=9.4)
        ax.set_ylabel(label, fontsize=10.6, color=ps.MUTED)
        ax.set_title(heading, fontsize=12, color=ps.INK, pad=10, loc="left")
        ps.clean(ax, grid_axis="y")

    panel(left, totals, ps.SLATE, "Total precipitation, millimetres",
          "Summed over the whole record", "{:,.0f}")
    panel(right, rates, ps.BLUE, "Precipitation per observed day, millimetres",
          "Divided by the days each month actually has", "{:.3f}")

    ps.title_block(
        fig, "February looks drier by total and is wetter per day",
        f"The same generated record, aggregated two ways. February is observed on "
        f"{example['february_observed_days']:,} days against January's "
        f"{example['january_observed_days']:,},\nso a raw total charges it for "
        f"{example['january_observed_days'] - example['february_observed_days']} "
        f"days it never had.")

    ps.footnote(fig, [
        f"By total February sits {abs(example['february_total_relative_to_january_pct'])}"
        f" percent below January, which is inside the width of the bars. Per observed "
        f"day it is "
        f"{example['february_rate_relative_to_january_pct']} percent above.",
        "These two are the only months whose order changes between the panels. "
        "The distortion applies to every February and to this record's December, "
        "but only here were the totals close enough for it to reorder anything.",
    ])
    ps.save(fig, FIGURES, "02_precipitation_by_month")


def figure_03_wind(summary: dict) -> None:
    """Is the windiest city windier, or luckier?"""
    wind = summary["wind_ranking"]
    cities = list(wind["per_city_mean_ms"])
    means = [wind["per_city_mean_ms"][city] for city in cities]

    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    fig.subplots_adjust(left=0.10, right=0.955, top=0.725, bottom=0.255)

    # One standard error either side of each mean, from the recorded values.
    error = wind["combined_standard_error_ms"] / np.sqrt(2)
    positions = np.arange(len(cities))
    ax.errorbar(positions, means, yerr=error * 1.96, fmt="o", markersize=9,
                color=ps.BLUE, ecolor=ps.SLATE_SOFT, elinewidth=2.4, capsize=7)
    for index, value in enumerate(means):
        ax.text(index, value + error * 2.6, f"{value:.3f}", ha="center",
                fontsize=10, color=ps.INK)

    ax.set_xticks(positions, cities, fontsize=10.4, color=ps.INK)
    ax.set_ylabel("Mean wind speed, metres per second", fontsize=11, color=ps.MUTED)
    ax.set_xlim(-0.6, len(cities) - 0.4)
    ps.clean(ax, grid_axis="y")

    ps.title_block(
        fig, "The windiest city is not windier than the next",
        "Mean wind speed per city with a 95 percent interval. The generator applies "
        "one wind formula to all six,\nwith no city term, so these are six draws "
        "from the same distribution.")

    ps.footnote(fig, [
        f"{wind['highest_mean']} leads {wind['second_highest']} by "
        f"{wind['difference_ms']:.4f} m/s, which is "
        f"{wind['difference_in_standard_errors']} standard errors. The full spread "
        f"across all six cities is",
        f"{wind['spread_across_all_cities_ms']:.4f} m/s. An earlier version of this "
        f"README listed the highest average wind speed as a feature of one city.",
    ])
    ps.save(fig, FIGURES, "03_wind_is_noise")


def figure_04_correlation(summary: dict, frame: pd.DataFrame) -> None:
    """How much of the temperature and humidity link is a pooling artifact?"""
    pair = summary["temperature_humidity"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 6.4),
                                      gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.075, right=0.965, top=0.70, bottom=0.245, wspace=0.26)

    sample = frame.sample(4000, random_state=42)
    for city, block in sample.groupby("city"):
        left.scatter(block["temp_mean_c"], block["humidity_pct"], s=6, alpha=0.20,
                     color=CITY_COLOUR[city], edgecolors="none", label=city)

    # Each city's own least-squares line, drawn only across that city's range,
    # against the pooled line drawn across all of them. The pooled line is
    # steeper than any of the six, which is the finding.
    for city, block in frame.groupby("city"):
        slope, intercept = np.polyfit(block["temp_mean_c"], block["humidity_pct"], 1)
        span = np.array([block["temp_mean_c"].quantile(0.01),
                         block["temp_mean_c"].quantile(0.99)])
        left.plot(span, slope * span + intercept, color=CITY_COLOUR[city],
                  linewidth=2.4, solid_capstyle="round")

    pooled_slope, pooled_intercept = np.polyfit(
        frame["temp_mean_c"], frame["humidity_pct"], 1)
    whole = np.array([frame["temp_mean_c"].quantile(0.005),
                      frame["temp_mean_c"].quantile(0.995)])
    left.plot(whole, pooled_slope * whole + pooled_intercept, color=ps.INK,
              linewidth=2.6, linestyle="--", dash_capstyle="round")

    left.set_xlabel("Daily mean temperature, degrees Celsius", fontsize=10.6,
                    color=ps.MUTED)
    left.set_ylabel("Relative humidity, percent", fontsize=10.6, color=ps.MUTED)
    left.set_title("The pooled line is steeper than any city's own", fontsize=12,
                   color=ps.INK, pad=10, loc="left")
    handles = [plt.Line2D([], [], color=ps.INK, linewidth=2.4, linestyle="--",
                          label="pooled")]
    handles += [plt.Line2D([], [], color=CITY_COLOUR[city], linewidth=2.4,
                           label=city) for city in sorted(CITY_COLOUR)]
    left.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
                labelcolor=ps.MUTED, handletextpad=0.5, handlelength=1.6,
                labelspacing=0.35)
    ps.clean(left, grid_axis="both")

    levels = ["pooled across\nall six cities", "within each\ncity",
              "within city\nand month"]
    values = [pair["pooled_across_cities"], pair["within_city_mean"],
              pair["within_city_and_month_mean"]]
    colours = [ps.RED, ps.AMBER, ps.BLUE]
    right.bar(np.arange(3), [abs(v) for v in values], width=0.55, color=colours)
    for index, value in enumerate(values):
        right.text(index, abs(value) + 0.018, f"{value:+.3f}", ha="center",
                   fontsize=11, color=ps.INK, fontweight="600")
    right.set_xticks(np.arange(3), levels, fontsize=9.6, color=ps.INK)
    right.set_ylabel("Correlation strength, absolute value", fontsize=10.6,
                     color=ps.MUTED)
    right.set_ylim(0, max(abs(v) for v in values) * 1.22)
    right.set_title("The same correlation, three ways", fontsize=12, color=ps.INK,
                    pad=10, loc="left")
    ps.clean(right, grid_axis="y")

    ps.title_block(
        fig, "Most of the correlation is between cities, not within them",
        "Temperature against humidity. Pooling six cities with different baselines "
        "mixes differences between\nthem into a relationship meant to describe days "
        "within one.")

    ps.footnote(fig, [
        f"Pooled the correlation reads {pair['pooled_across_cities']:+.3f}. Computed "
        f"inside each city it is {pair['within_city_mean']:+.3f}, and inside each "
        f"city and month {pair['within_city_and_month_mean']:+.3f}. The same holds "
        f"for the slope: pooled it is",
        f"{pair['pooled_slope']:+.3f} percent humidity per degree, against "
        f"{pair['steepest_city_slope']:+.3f} for the steepest single city. The "
        f"correlation matrix in the original analysis reported the pooled version.",
    ])
    ps.save(fig, FIGURES, "04_correlation_pooling")


def main() -> int:
    ps.apply()
    summary, frame = load()
    print(f"  reading {SUMMARY.relative_to(ROOT).as_posix()}")
    figure_01_annual_cycle(summary, frame)
    figure_02_precipitation(summary, frame)
    figure_03_wind(summary)
    figure_04_correlation(summary, frame)
    print(f"  figures in {FIGURES.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
