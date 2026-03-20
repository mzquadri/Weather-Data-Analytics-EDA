# Weather Data Analytics — Exploratory Data Analysis

Comprehensive exploratory data analysis of global weather data across 6 cities and different climate zones. This project demonstrates data wrangling, statistical analysis, and professional data visualization techniques using Python.

## Dataset

Synthetic 10-year daily weather data (2014-2023) for 6 cities:

| City | Climate Zone | Avg Temp (°C) | Key Feature |
|------|-------------|---------------|-------------|
| Munich | Temperate Continental | ~8.5 | Cold winters, warm summers |
| Delhi | Subtropical | ~25.0 | Monsoon precipitation pattern |
| Tokyo | Humid Subtropical | ~15.5 | Highest total precipitation |
| New York | Humid Continental | ~12.5 | Highest average wind speed |
| Sydney | Oceanic | ~18.0 | Reversed seasons (Southern Hemisphere) |
| Nairobi | Tropical Highland | ~18.5 | Minimal temperature variation |

**Features**: Temperature (mean/max/min), precipitation, humidity, wind speed, pressure, cloud cover, UV index

**Total records**: 21,900 (6 cities x 3,650 days)

## Key Visualizations

### Temperature Distributions

![Temperature Violin](results/figures/01_temp_violin.png)

### Monthly Temperature Heatmap

![Temperature Heatmap](results/figures/02_temp_heatmap.png)

### Seasonal Temperature Comparison

![Seasonal Temperature](results/figures/03_seasonal_temp.png)

### Monthly Precipitation Patterns

![Precipitation Patterns](results/figures/04_precip_monthly.png)

### Diurnal Temperature Range

![Temperature Range](results/figures/05_temp_range.png)

### Feature Correlation Matrix

![Correlation Matrix](results/figures/06_correlation_matrix.png)

### Wind Speed Distributions

![Wind Distributions](results/figures/07_wind_distributions.png)

### Temperature Anomaly Trends

![Temperature Anomaly](results/figures/08_temp_anomaly.png)

### Precipitation Exceedance Curves

![Precipitation Exceedance](results/figures/09_precip_exceedance.png)

### Temperature vs Humidity

![Temp Humidity Scatter](results/figures/10_temp_humidity_scatter.png)

### UV Index Seasonal Patterns

![UV Index](results/figures/11_uv_index.png)

### Pairwise Feature Relationships

![Pairplot](results/figures/12_pairplot.png)

## Project Structure

```
.
├── src/
│   ├── generate_data.py     # Synthetic weather data generation
│   └── eda_analysis.py      # Full EDA pipeline (12 visualizations)
├── data/
│   └── weather_data.csv     (generated, not tracked)
├── results/
│   ├── figures/             # 12 publication-quality plots
│   ├── key_findings.json    # Automated key findings
│   └── city_summary_statistics.csv
├── requirements.txt
└── README.md
```

## Key Findings

- **Hottest city**: Delhi (mean ~25°C)
- **Coldest city**: Munich (mean ~8.5°C)
- **Wettest city**: Tokyo (highest total precipitation)
- **Windiest city**: New York (highest mean wind speed)
- **Delhi monsoon**: Clear precipitation spike in July-September
- **Southern Hemisphere**: Sydney shows reversed seasonal patterns as expected
- **Nairobi**: Near-equatorial location results in minimal seasonal temperature variation
- Strong negative correlation between temperature and pressure
- Strong positive correlation between cloud cover and humidity

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/eda_analysis.py
```

## Tech Stack

- **pandas** — Data manipulation and aggregation
- **matplotlib** — Base plotting library
- **seaborn** — Statistical visualizations
- **NumPy** — Numerical computations
- **SciPy** — Statistical functions

## License

MIT
