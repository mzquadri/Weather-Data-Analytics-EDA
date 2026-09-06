"""Tests for the generated record: schema, coverage, units and season labels.

The analytical invariants worth guarding here are the ones a reader would assume
without checking. That the record covers the dates it says it does, that the
hemispheres are the right way round, and that each column stays inside the range
its unit permits.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.generate_data import generate_weather_data
from src.summary import MEASUREMENTS, coverage, schema

#: Ranges each unit permits. A value outside one of these is not an extreme, it
#: is a bug: humidity above 100 percent or a UV index of 20 is not a reading.
PERMITTED = {
    "temp_mean_c": (-60, 60),
    "temp_max_c": (-60, 65),
    "temp_min_c": (-70, 60),
    "precipitation_mm": (0, 500),
    "humidity_pct": (0, 100),
    "wind_speed_ms": (0, 120),
    "pressure_hpa": (850, 1100),
    "cloud_cover_pct": (0, 100),
    "uv_index": (0, 12),
}


class Schema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = generate_weather_data(n_years=3, seed=5)

    def test_every_documented_measurement_is_present(self):
        for name in MEASUREMENTS:
            with self.subTest(column=name):
                self.assertIn(name, self.frame.columns)

    def test_the_identifying_columns_are_present(self):
        for name in ("date", "city", "latitude", "month", "year", "season"):
            with self.subTest(column=name):
                self.assertIn(name, self.frame.columns)

    def test_every_measurement_is_numeric(self):
        for name in MEASUREMENTS:
            with self.subTest(column=name):
                self.assertTrue(np.issubdtype(self.frame[name].dtype, np.number))

    def test_nothing_is_missing(self):
        self.assertEqual(int(self.frame.isna().sum().sum()), 0)

    def test_the_schema_summary_covers_every_measurement(self):
        described = {entry["name"] for entry in schema(self.frame)}
        self.assertEqual(described, set(MEASUREMENTS))


class Units(unittest.TestCase):
    """Values must stay inside what their unit permits."""

    @classmethod
    def setUpClass(cls):
        cls.frame = generate_weather_data(n_years=3, seed=6)

    def test_no_value_falls_outside_its_permitted_range(self):
        for name, (low, high) in PERMITTED.items():
            with self.subTest(column=name):
                self.assertGreaterEqual(float(self.frame[name].min()), low)
                self.assertLessEqual(float(self.frame[name].max()), high)

    def test_daily_maximum_is_never_below_daily_minimum(self):
        self.assertTrue((self.frame["temp_max_c"] >= self.frame["temp_min_c"]).all())

    def test_the_daily_mean_sits_between_the_maximum_and_the_minimum(self):
        self.assertTrue((self.frame["temp_mean_c"] <= self.frame["temp_max_c"]).all())
        self.assertTrue((self.frame["temp_mean_c"] >= self.frame["temp_min_c"]).all())

    def test_precipitation_is_never_negative(self):
        self.assertGreaterEqual(float(self.frame["precipitation_mm"].min()), 0.0)


class Coverage(unittest.TestCase):
    def test_the_record_is_two_days_short_of_ten_calendar_years(self):
        """The leap-day shortfall the README describes.

        n_years * 365 ignores 2016 and 2020, so a ten-year request ends on
        29 December 2023 rather than 31 December.
        """
        frame = generate_weather_data(n_years=10, seed=42)
        found = coverage(frame)
        self.assertEqual(found["first_date"], "2014-01-01")
        self.assertEqual(found["last_date"], "2023-12-29")
        self.assertEqual(found["years_short_of_a_full_calendar_year"], {2023: 2})

    def test_there_are_no_gaps_or_duplicates(self):
        frame = generate_weather_data(n_years=4, seed=7)
        found = coverage(frame)
        self.assertEqual(found["largest_gap_between_consecutive_days"], 1)
        self.assertEqual(found["duplicate_city_date_pairs"], 0)

    def test_every_city_covers_the_same_days(self):
        frame = generate_weather_data(n_years=3, seed=8)
        per_city = frame.groupby("city")["date"].agg(["min", "max", "count"])
        self.assertEqual(per_city["count"].nunique(), 1)
        self.assertEqual(per_city["min"].nunique(), 1)
        self.assertEqual(per_city["max"].nunique(), 1)


class Hemispheres(unittest.TestCase):
    """Season labels must follow the sign of the latitude, not the calendar."""

    @classmethod
    def setUpClass(cls):
        cls.frame = generate_weather_data(n_years=5, seed=9)

    def test_january_is_winter_in_the_north_and_summer_in_the_south(self):
        january = self.frame[self.frame["month"] == 1]
        north = january[january["latitude"] >= 0]["season"].unique()
        south = january[january["latitude"] < 0]["season"].unique()
        self.assertEqual(set(north), {"Winter"})
        self.assertEqual(set(south), {"Summer"})

    def test_july_is_the_other_way_round(self):
        july = self.frame[self.frame["month"] == 7]
        north = july[july["latitude"] >= 0]["season"].unique()
        south = july[july["latitude"] < 0]["season"].unique()
        self.assertEqual(set(north), {"Summer"})
        self.assertEqual(set(south), {"Winter"})

    def test_the_temperature_cycle_follows_the_label(self):
        """The label and the values must agree, or one of them is wrong.

        A hemisphere-aware label is only useful if the days it marks Summer are
        the warm ones. This holds in both hemispheres precisely because the label
        is derived from the latitude rather than from the calendar.
        """
        for city, block in self.frame.groupby("city"):
            with self.subTest(city=city):
                by_season = block.groupby("season")["temp_mean_c"].mean()
                self.assertGreater(by_season["Summer"], by_season["Winter"])

    def test_southern_cities_peak_in_the_northern_winter_months(self):
        southern = self.frame[self.frame["latitude"] < 0]
        monthly = southern.groupby("month")["temp_mean_c"].mean()
        self.assertIn(int(monthly.idxmax()), {12, 1, 2})
        self.assertIn(int(monthly.idxmin()), {6, 7, 8})


class Reproducibility(unittest.TestCase):
    def test_the_same_seed_gives_the_same_record(self):
        first = generate_weather_data(n_years=1, seed=3)
        second = generate_weather_data(n_years=1, seed=3)
        self.assertTrue(first.equals(second))

    def test_a_different_seed_gives_a_different_record(self):
        first = generate_weather_data(n_years=1, seed=3)
        second = generate_weather_data(n_years=1, seed=4)
        self.assertFalse(first.equals(second))


if __name__ == "__main__":
    unittest.main()
