"""
Comprehensive Exploratory Data Analysis of Weather Data.

Generates 10+ professional publication-quality visualizations covering:
- Temperature distributions and trends
- Precipitation patterns and extremes
- Seasonal and geographic comparisons
- Correlation analysis
- Climate anomaly detection
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# Style configuration
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#009688"]


def run_eda(csv_path: str, output_dir: str = "../results"):
    """Run full EDA pipeline and save all figures and statistics."""

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    cities = sorted(df["city"].unique())
    n_cities = len(cities)

    print(f"Dataset: {len(df)} records, {n_cities} cities")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # ===============================================================
    # 1. DATASET OVERVIEW — Summary Statistics
    # ===============================================================
    summary = (
        df.groupby("city")
        .agg(
            {
                "temp_mean_c": ["mean", "std", "min", "max"],
                "precipitation_mm": ["mean", "sum", "max"],
                "humidity_pct": ["mean", "std"],
                "wind_speed_ms": ["mean", "max"],
            }
        )
        .round(2)
    )
    summary.to_csv(os.path.join(output_dir, "city_summary_statistics.csv"))
    print("\nCity-level summary saved.")

    # ===============================================================
    # 2. TEMPERATURE DISTRIBUTIONS BY CITY (Violin + Box)
    # ===============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x="city",
        y="temp_mean_c",
        palette=PALETTE,
        inner="box",
        order=cities,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.set_title("Temperature Distribution by City")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "01_temp_violin.png"), dpi=150)
    plt.close(fig)
    print("Saved: 01_temp_violin.png")

    # ===============================================================
    # 3. MONTHLY TEMPERATURE HEATMAP
    # ===============================================================
    monthly_temp = df.groupby(["city", "month"])["temp_mean_c"].mean().unstack()
    monthly_temp = monthly_temp.reindex(cities)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        monthly_temp,
        annot=True,
        fmt=".1f",
        cmap="RdYlBu_r",
        center=15,
        linewidths=0.5,
        ax=ax,
        xticklabels=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    ax.set_ylabel("")
    ax.set_title("Monthly Mean Temperature (°C) by City")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_temp_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved: 02_temp_heatmap.png")

    # ===============================================================
    # 4. SEASONAL TEMPERATURE COMPARISON (Box plots)
    # ===============================================================
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    for idx, city in enumerate(cities):
        ax = axes[idx // 3, idx % 3]
        city_data = df[df["city"] == city]
        sns.boxplot(
            data=city_data,
            x="season",
            y="temp_mean_c",
            order=season_order,
            palette="coolwarm",
            ax=ax,
        )
        ax.set_title(city, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Temperature (°C)" if idx % 3 == 0 else "")
    fig.suptitle(
        "Seasonal Temperature Distributions", fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "03_seasonal_temp.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved: 03_seasonal_temp.png")

    # ===============================================================
    # 5. PRECIPITATION PATTERNS — Monthly totals
    # ===============================================================
    monthly_precip = (
        df.groupby(["city", "month"])["precipitation_mm"].sum().reset_index()
    )
    monthly_precip["month_name"] = monthly_precip["month"].map(
        {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, city in enumerate(cities):
        city_data = monthly_precip[monthly_precip["city"] == city]
        ax.plot(
            city_data["month"],
            city_data["precipitation_mm"],
            marker="o",
            linewidth=2,
            color=PALETTE[i],
            label=city,
        )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    ax.set_ylabel("Total Precipitation (mm)")
    ax.set_title("Monthly Precipitation Totals by City (10-Year Sum)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_precip_monthly.png"), dpi=150)
    plt.close(fig)
    print("Saved: 04_precip_monthly.png")

    # ===============================================================
    # 6. DAILY TEMPERATURE RANGE (Diurnal variation)
    # ===============================================================
    df["temp_range"] = df["temp_max_c"] - df["temp_min_c"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="city", y="temp_range", palette=PALETTE, order=cities, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Daily Temperature Range (°C)")
    ax.set_title("Diurnal Temperature Range by City")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "05_temp_range.png"), dpi=150)
    plt.close(fig)
    print("Saved: 05_temp_range.png")

    # ===============================================================
    # 7. CORRELATION MATRIX (per city, combined)
    # ===============================================================
    numeric_cols = [
        "temp_mean_c",
        "precipitation_mm",
        "humidity_pct",
        "wind_speed_ms",
        "pressure_hpa",
        "cloud_cover_pct",
        "uv_index",
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        xticklabels=[c.replace("_", "\n") for c in numeric_cols],
        yticklabels=[c.replace("_", "\n") for c in numeric_cols],
    )
    ax.set_title("Feature Correlation Matrix (All Cities)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "06_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: 06_correlation_matrix.png")

    # ===============================================================
    # 8. WIND ROSE — Wind speed distributions
    # ===============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, city in enumerate(cities):
        ax = axes[idx // 3, idx % 3]
        city_data = df[df["city"] == city]["wind_speed_ms"]
        ax.hist(city_data, bins=40, color=PALETTE[idx], edgecolor="white", alpha=0.8)
        ax.axvline(
            city_data.mean(),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {city_data.mean():.1f} m/s",
        )
        ax.set_title(city, fontweight="bold")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
    fig.suptitle("Wind Speed Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "07_wind_distributions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: 07_wind_distributions.png")

    # ===============================================================
    # 9. TEMPERATURE ANOMALY OVER TIME
    # ===============================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, city in enumerate(cities):
        city_df = df[df["city"] == city].copy()
        # Calculate monthly anomaly relative to long-term monthly mean
        monthly_means = city_df.groupby("month")["temp_mean_c"].transform("mean")
        city_df["anomaly"] = city_df["temp_mean_c"] - monthly_means
        # Rolling 90-day mean of anomaly
        rolling_anomaly = (
            city_df.set_index("date")["anomaly"].rolling(90, center=True).mean()
        )
        ax.plot(
            rolling_anomaly.index,
            rolling_anomaly.values,
            linewidth=1.2,
            color=PALETTE[i],
            label=city,
            alpha=0.8,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature Anomaly (°C)")
    ax.set_title("90-Day Rolling Temperature Anomaly (Relative to Monthly Climatology)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "08_temp_anomaly.png"), dpi=150)
    plt.close(fig)
    print("Saved: 08_temp_anomaly.png")

    # ===============================================================
    # 10. EXTREME EVENTS — Precipitation exceedance
    # ===============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, city in enumerate(cities):
        city_precip = df[df["city"] == city]["precipitation_mm"]
        city_precip = city_precip[city_precip > 0].sort_values(ascending=False).values
        exceedance = np.arange(1, len(city_precip) + 1) / len(city_precip)
        ax.semilogy(
            city_precip, exceedance, linewidth=1.5, color=PALETTE[i], label=city
        )
    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_ylabel("Exceedance Probability")
    ax.set_title("Precipitation Exceedance Probability Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "09_precip_exceedance.png"), dpi=150)
    plt.close(fig)
    print("Saved: 09_precip_exceedance.png")

    # ===============================================================
    # 11. HUMIDITY vs TEMPERATURE — Scatter with density
    # ===============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, city in enumerate(cities):
        ax = axes[idx // 3, idx % 3]
        city_data = df[df["city"] == city]
        scatter = ax.scatter(
            city_data["temp_mean_c"],
            city_data["humidity_pct"],
            c=city_data["precipitation_mm"],
            cmap="Blues",
            s=3,
            alpha=0.4,
            vmin=0,
            vmax=20,
        )
        ax.set_title(city, fontweight="bold")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Humidity (%)")
    fig.suptitle(
        "Temperature vs Humidity (color = precipitation)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.6, label="Precipitation (mm)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "10_temp_humidity_scatter.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: 10_temp_humidity_scatter.png")

    # ===============================================================
    # 12. UV INDEX — Seasonal patterns
    # ===============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_uv = df.groupby(["city", "month"])["uv_index"].mean().reset_index()
    for i, city in enumerate(cities):
        city_uv = monthly_uv[monthly_uv["city"] == city]
        ax.plot(
            city_uv["month"],
            city_uv["uv_index"],
            marker="s",
            linewidth=2,
            color=PALETTE[i],
            label=city,
        )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    ax.set_ylabel("UV Index")
    ax.set_title("Monthly Mean UV Index by City")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.axhline(6, color="orange", linestyle="--", alpha=0.5, label="High UV Threshold")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "11_uv_index.png"), dpi=150)
    plt.close(fig)
    print("Saved: 11_uv_index.png")

    # ===============================================================
    # 13. PAIR PLOT (sampled for performance)
    # ===============================================================
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    pair_cols = ["temp_mean_c", "humidity_pct", "wind_speed_ms", "uv_index", "city"]
    g = sns.pairplot(
        sample[pair_cols],
        hue="city",
        palette=PALETTE,
        plot_kws={"alpha": 0.3, "s": 8},
        diag_kws={"alpha": 0.5},
        height=2.2,
        aspect=1,
    )
    g.fig.suptitle(
        "Pairwise Feature Relationships", y=1.02, fontsize=14, fontweight="bold"
    )
    g.savefig(os.path.join(figures_dir, "12_pairplot.png"), dpi=120)
    plt.close(g.fig)
    print("Saved: 12_pairplot.png")

    # ===============================================================
    # SAVE KEY FINDINGS
    # ===============================================================
    findings = {
        "n_records": len(df),
        "n_cities": n_cities,
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "hottest_city": df.groupby("city")["temp_mean_c"].mean().idxmax(),
        "coldest_city": df.groupby("city")["temp_mean_c"].mean().idxmin(),
        "wettest_city": df.groupby("city")["precipitation_mm"].sum().idxmax(),
        "windiest_city": df.groupby("city")["wind_speed_ms"].mean().idxmax(),
        "max_temp_recorded": float(df["temp_max_c"].max()),
        "min_temp_recorded": float(df["temp_min_c"].min()),
        "max_daily_precip": float(df["precipitation_mm"].max()),
        "total_figures_generated": 12,
    }
    with open(os.path.join(output_dir, "key_findings.json"), "w") as f:
        json.dump(findings, f, indent=2)

    print(f"\nKey findings:")
    for k, v in findings.items():
        print(f"  {k}: {v}")

    return findings


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "weather_data.csv"
    )
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    run_eda(csv_path=data_path, output_dir=results_dir)
