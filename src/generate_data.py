"""
Generate synthetic multi-city weather dataset for EDA.

Creates a realistic 10-year daily weather dataset for 6 cities across
different climate zones, with proper seasonal patterns, correlations
between variables, and realistic extreme events.
"""

import numpy as np
import pandas as pd
import os


def generate_weather_data(n_years: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily weather data for multiple cities."""
    rng = np.random.RandomState(seed)

    cities = {
        "Munich": {
            "lat": 48.14,
            "base_temp": 8.5,
            "amp": 12,
            "precip_scale": 2.8,
            "humid_base": 72,
        },
        "Delhi": {
            "lat": 28.61,
            "base_temp": 25.0,
            "amp": 10,
            "precip_scale": 3.5,
            "humid_base": 55,
        },
        "Tokyo": {
            "lat": 35.68,
            "base_temp": 15.5,
            "amp": 13,
            "precip_scale": 4.0,
            "humid_base": 68,
        },
        "New York": {
            "lat": 40.71,
            "base_temp": 12.5,
            "amp": 14,
            "precip_scale": 3.0,
            "humid_base": 65,
        },
        "Sydney": {
            "lat": -33.87,
            "base_temp": 18.0,
            "amp": 6,
            "precip_scale": 2.5,
            "humid_base": 62,
        },
        "Nairobi": {
            "lat": -1.29,
            "base_temp": 18.5,
            "amp": 2,
            "precip_scale": 2.0,
            "humid_base": 58,
        },
    }

    n_days = n_years * 365
    dates = pd.date_range(start="2014-01-01", periods=n_days, freq="D")

    all_records = []

    for city, params in cities.items():
        doy = dates.dayofyear.values.astype(float)
        # Reverse seasons for southern hemisphere
        phase_shift = 0 if params["lat"] > 0 else 182

        # Temperature
        seasonal_temp = params["base_temp"] + params["amp"] * np.sin(
            2 * np.pi * (doy - 100 + phase_shift) / 365
        )
        daily_temp = seasonal_temp + rng.normal(0, 2.5, n_days)
        temp_max = daily_temp + rng.uniform(3, 8, n_days)
        temp_min = daily_temp - rng.uniform(3, 8, n_days)

        # Precipitation
        precip_prob = 0.3 + 0.15 * np.sin(2 * np.pi * (doy - 50 + phase_shift) / 365)
        # Delhi: monsoon pattern (July-September)
        if city == "Delhi":
            monsoon = ((doy > 180) & (doy < 270)).astype(float)
            precip_prob = 0.15 + 0.55 * monsoon
        rain_occur = rng.binomial(1, np.clip(precip_prob, 0.05, 0.7))
        rain_amount = rng.exponential(params["precip_scale"], n_days) * rain_occur
        # Extreme events
        extreme = rng.binomial(1, 0.003, n_days).astype(bool)
        rain_amount[extreme] *= rng.uniform(5, 12, extreme.sum())

        # Humidity (correlated with temp inversely and precip positively)
        humidity = (
            params["humid_base"]
            - 0.5 * (daily_temp - params["base_temp"])
            + 8 * rain_occur
        )
        humidity += rng.normal(0, 5, n_days)
        humidity = np.clip(humidity, 10, 100)

        # Wind speed (m/s) — higher in winter
        wind_base = 3.5 - 1.0 * np.sin(2 * np.pi * (doy - 100 + phase_shift) / 365)
        wind_speed = rng.weibull(2.0, n_days) * wind_base
        wind_speed = np.clip(wind_speed, 0.2, 25)

        # Pressure (hPa) — inversely related to temperature
        pressure = 1013.25 - 0.3 * (daily_temp - 15) + rng.normal(0, 5, n_days)

        # Cloud cover (0-100%)
        cloud_cover = 20 + 30 * rain_occur + 0.3 * humidity + rng.normal(0, 10, n_days)
        cloud_cover = np.clip(cloud_cover, 0, 100)

        # UV Index (0-12)
        solar_angle = np.clip(np.sin(2 * np.pi * (doy - 80 + phase_shift) / 365), 0, 1)
        uv_index = (
            11
            * solar_angle
            * (1 - 0.7 * cloud_cover / 100)
            * (1 - 0.3 * abs(params["lat"]) / 60)
        )
        uv_index = np.clip(uv_index + rng.normal(0, 0.5, n_days), 0, 12)

        for i in range(n_days):
            all_records.append(
                {
                    "date": dates[i],
                    "city": city,
                    "latitude": params["lat"],
                    "temp_mean_c": round(daily_temp[i], 1),
                    "temp_max_c": round(temp_max[i], 1),
                    "temp_min_c": round(temp_min[i], 1),
                    "precipitation_mm": round(max(0, rain_amount[i]), 1),
                    "humidity_pct": round(humidity[i], 1),
                    "wind_speed_ms": round(wind_speed[i], 1),
                    "pressure_hpa": round(pressure[i], 1),
                    "cloud_cover_pct": round(cloud_cover[i], 1),
                    "uv_index": round(max(0, uv_index[i]), 1),
                }
            )

    df = pd.DataFrame(all_records)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    northern_seasons = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    southern_seasons = {
        12: "Summer",
        1: "Summer",
        2: "Summer",
        3: "Autumn",
        4: "Autumn",
        5: "Autumn",
        6: "Winter",
        7: "Winter",
        8: "Winter",
        9: "Spring",
        10: "Spring",
        11: "Spring",
    }
    df["season"] = [
        (northern_seasons if latitude >= 0 else southern_seasons)[month]
        for latitude, month in zip(df["latitude"], df["month"])
    ]

    return df


if __name__ == "__main__":
    df = generate_weather_data(n_years=10)
    save_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "weather_data.csv"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Generated {len(df)} records for {df['city'].nunique()} cities")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nCities: {df['city'].unique().tolist()}")
    print(f"\n{df.describe().round(2)}")
