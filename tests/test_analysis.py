"""Tests for the analysis itself: aggregation, pooling, ranking and extremes.

Each test here guards one of the mistakes the README describes. The point is not
that the functions run, it is that they still separate the things that were
previously conflated: a total from a rate, a between-city difference from a
within-city one, and a lead from the noise around it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.generate_data import generate_weather_data
from src.summary import (
    MEASUREMENTS,
    build,
    correlation_levels,
    extremes,
    monthly_precipitation,
    schema,
    uv_floor,
    wind_is_not_a_city_difference,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "results" / "summary.json"


class Aggregation(unittest.TestCase):
    """Totals and rates must not be confused with each other."""

    def test_the_rate_is_the_total_divided_by_the_days_observed(self):
        frame = generate_weather_data(n_years=2, seed=11)
        found = monthly_precipitation(frame)
        for month in range(1, 13):
            with self.subTest(month=month):
                total = found["total_mm_by_month"][month]
                days = found["observations_by_month"][month]
                self.assertAlmostEqual(
                    found["mm_per_day_by_month"][month], total / days, places=2)

    def test_the_totals_add_up_to_the_whole_record(self):
        frame = generate_weather_data(n_years=2, seed=12)
        found = monthly_precipitation(frame)
        self.assertAlmostEqual(sum(found["total_mm_by_month"].values()),
                               float(frame["precipitation_mm"].sum()), places=0)

    def test_a_shorter_month_can_lead_on_rate_and_lose_on_total(self):
        """The exact confusion the README describes, on values chosen to force it.

        Two months, one with 31 days at 1.0 mm and one with 28 days at 1.1 mm.
        The shorter month rains more per day and still totals less.
        """
        rows = ([{"month": 1, "precipitation_mm": 1.0}] * 31
                + [{"month": 2, "precipitation_mm": 1.1}] * 28)
        found = monthly_precipitation(pd.DataFrame(rows))
        self.assertLess(found["total_mm_by_month"][2], found["total_mm_by_month"][1])
        self.assertGreater(found["mm_per_day_by_month"][2],
                           found["mm_per_day_by_month"][1])
        self.assertEqual(found["wettest_month_by_total"], 1)
        self.assertEqual(found["wettest_month_by_rate"], 2)

    def test_precipitation_is_the_only_column_documented_as_summing(self):
        """Guards against averaging an accumulation or accumulating an average."""
        frame = generate_weather_data(n_years=1, seed=13)
        by_name = {entry["name"]: entry["aggregates_by"] for entry in schema(frame)}
        self.assertEqual(by_name["precipitation_mm"], "sum")
        summing = {name for name, how in by_name.items() if how == "sum"}
        self.assertEqual(summing, {"precipitation_mm"})
        self.assertEqual(len(by_name), len(MEASUREMENTS))


class Pooling(unittest.TestCase):
    """The three correlation levels must actually be three different things."""

    @staticmethod
    def simpson_frame() -> pd.DataFrame:
        """Perfectly positive inside each city, negative once pooled.

        City A is cool and humid, city B warm and dry. Inside each, humidity
        rises exactly with temperature.
        """
        rows = []
        for city, offset, intercept in (("A", 0, 80), ("B", 20, 40)):
            for step in range(11):
                temp = offset + step
                rows.append({"city": city, "month": 1, "temp_mean_c": float(temp),
                             "humidity_pct": float(temp + intercept)})
        return pd.DataFrame(rows)

    def test_pooling_can_reverse_the_sign_of_the_within_city_relationship(self):
        found = correlation_levels(self.simpson_frame(), "temp_mean_c", "humidity_pct")
        self.assertAlmostEqual(found["within_city_mean"], 1.0, places=6)
        self.assertAlmostEqual(found["within_city_and_month_mean"], 1.0, places=6)
        self.assertLess(found["pooled_across_cities"], 0.0)

    def test_month_is_a_further_level_below_city(self):
        """With one month per city the two within-city levels must coincide."""
        found = correlation_levels(self.simpson_frame(), "temp_mean_c", "humidity_pct")
        self.assertEqual(found["within_city_mean"],
                         found["within_city_and_month_mean"])

    def test_the_pooled_value_matches_a_plain_correlation(self):
        frame = generate_weather_data(n_years=2, seed=14)
        found = correlation_levels(frame, "temp_mean_c", "humidity_pct")
        plain = float(np.corrcoef(frame["temp_mean_c"], frame["humidity_pct"])[0, 1])
        self.assertAlmostEqual(found["pooled_across_cities"], round(plain, 3), places=6)

    def test_the_pooled_slope_is_steeper_than_every_city_taken_alone(self):
        """The claim the left panel of figure 04 makes.

        Pooling clusters that sit at different heights produces a line steeper
        than any of them, which is why the pooled coefficient overstates the
        within-city relationship.
        """
        frame = generate_weather_data(n_years=10, seed=42)
        found = correlation_levels(frame, "temp_mean_c", "humidity_pct")
        self.assertLess(found["pooled_slope"], min(found["per_city_slope"].values()))
        self.assertEqual(found["steepest_city_slope"],
                         min(found["per_city_slope"].values()))
        self.assertEqual(found["shallowest_city_slope"],
                         max(found["per_city_slope"].values()))

    def test_the_per_city_values_average_to_the_reported_within_city_mean(self):
        frame = generate_weather_data(n_years=2, seed=15)
        found = correlation_levels(frame, "temp_mean_c", "humidity_pct")
        self.assertAlmostEqual(found["within_city_mean"],
                               float(np.mean(list(found["per_city"].values()))),
                               places=2)


class WindRanking(unittest.TestCase):
    """A lead has to be reported against the noise around it."""

    def test_six_identical_distributions_give_a_lead_under_two_standard_errors(self):
        """The generator has no city term for wind, so this must stay small."""
        frame = generate_weather_data(n_years=10, seed=42)
        found = wind_is_not_a_city_difference(frame)
        self.assertLess(found["difference_in_standard_errors"], 2.0)

    def test_a_real_city_effect_is_reported_as_many_standard_errors(self):
        """If wind ever gained a city term, the measure must say so."""
        rng = np.random.default_rng(0)
        rows = []
        for city, offset in (("A", 0.0), ("B", 2.0)):
            for value in rng.normal(5.0 + offset, 1.0, 2000):
                rows.append({"city": city, "wind_speed_ms": float(value)})
        found = wind_is_not_a_city_difference(pd.DataFrame(rows))
        self.assertEqual(found["highest_mean"], "B")
        self.assertGreater(found["difference_in_standard_errors"], 20.0)

    def test_the_combined_standard_error_is_the_root_sum_of_squares(self):
        frame = generate_weather_data(n_years=3, seed=16)
        found = wind_is_not_a_city_difference(frame)
        stats = frame.groupby("city")["wind_speed_ms"].agg(["std", "count"])
        errors = stats["std"] / np.sqrt(stats["count"])
        expected = float(np.sqrt(errors[found["highest_mean"]] ** 2
                                 + errors[found["second_highest"]] ** 2))
        self.assertAlmostEqual(found["combined_standard_error_ms"],
                               round(expected, 4), places=6)

    def test_the_spread_covers_the_first_and_last_of_the_ranking(self):
        frame = generate_weather_data(n_years=3, seed=17)
        found = wind_is_not_a_city_difference(frame)
        means = list(found["per_city_mean_ms"].values())
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertAlmostEqual(found["spread_across_all_cities_ms"],
                               round(means[0] - means[-1], 4), places=2)


class Extremes(unittest.TestCase):
    """Selection is by rule, so the rule is what gets tested."""

    @staticmethod
    def hand_made() -> pd.DataFrame:
        rain = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        return pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "city": ["A"] * 10,
            "precipitation_mm": rain,
            "temp_max_c": np.arange(10.0, 20.0),
            "temp_min_c": np.arange(0.0, 10.0),
            "wind_speed_ms": np.arange(1.0, 11.0),
        })

    def test_the_longest_dry_run_is_the_longest_run_and_not_the_total(self):
        """Eight dry days in total, but the longest unbroken run is four."""
        found = extremes(self.hand_made())
        self.assertEqual(found["longest_dry_runs"][0]["longest_dry_run_days"], 4)

    def test_each_extreme_comes_from_its_own_column(self):
        found = extremes(self.hand_made())
        self.assertEqual(found["hottest_day"]["date"], "2020-01-10")
        self.assertEqual(found["coldest_day"]["date"], "2020-01-01")
        self.assertEqual(found["wettest_day"]["date"], "2020-01-04")
        self.assertEqual(found["windiest_day"]["date"], "2020-01-10")

    def test_the_dry_runs_are_ordered_longest_first(self):
        frame = generate_weather_data(n_years=3, seed=18)
        runs = [entry["longest_dry_run_days"] for entry in
                extremes(frame)["longest_dry_runs"]]
        self.assertEqual(runs, sorted(runs, reverse=True))


class UvFloor(unittest.TestCase):
    def test_the_floor_is_reported_for_every_city(self):
        frame = generate_weather_data(n_years=3, seed=19)
        found = uv_floor(frame)["share_of_days_at_or_below_0.05"]
        self.assertEqual(set(found), set(frame["city"].unique()))
        for city, share in found.items():
            with self.subTest(city=city):
                self.assertGreater(share, 0.0)
                self.assertLess(share, 0.5)


class RecordedSummary(unittest.TestCase):
    """The committed artifact must still be what the code produces."""

    @classmethod
    def setUpClass(cls):
        if not RECORDED.is_file():
            raise unittest.SkipTest("results/summary.json has not been generated")
        cls.recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
        cls.fresh = json.loads(json.dumps(build()))

    def test_the_coverage_is_unchanged(self):
        self.assertEqual(self.fresh["coverage"], self.recorded["coverage"])

    def test_the_wind_ranking_is_unchanged(self):
        self.assertEqual(self.fresh["wind_ranking"], self.recorded["wind_ranking"])

    def test_the_correlation_levels_are_unchanged(self):
        for key in ("temperature_humidity", "temperature_pressure"):
            with self.subTest(pair=key):
                self.assertEqual(self.fresh[key], self.recorded[key])

    def test_the_extremes_are_unchanged(self):
        self.assertEqual(self.fresh["extremes"], self.recorded["extremes"])

    def test_the_city_statistics_are_unchanged(self):
        self.assertEqual(self.fresh["cities"], self.recorded["cities"])

    def test_the_monthly_precipitation_is_unchanged(self):
        self.assertEqual(self.fresh["monthly_precipitation"],
                         self.recorded["monthly_precipitation"])

    def test_the_schema_is_unchanged(self):
        self.assertEqual(self.fresh["schema"], self.recorded["schema"])


if __name__ == "__main__":
    unittest.main()
