# 🌫️ Air Pollution Analysis & Forecasting

## 📌 Overview

An end-to-end air pollution data science project covering **exploratory data analysis (EDA) in Python** and **time-series forecasting in R**, applied to real-world pollution datasets from Aarhus, Denmark.

---

## 📁 Project Structure

```
Air Pollution Data Analysis/
│── Air Pollution Data Analysis.py  # EDA pipeline (Python)
│── TimeSeriesTR.R                  # ARIMA training & forecasting (R)
│── TimeSeriesTS.R                  # ARIMA testing & evaluation (R)
│── PollutionData190100.csv
│── PollutionData209960.csv
└── README.md
```

---

## 🔬 Part 1 — EDA (Python)

**Goal:** Understand pollution patterns, distributions, and correlations across two datasets.

- Missing value detection, statistical summaries, feature inspection
- Correlation heatmaps, pair plots, histograms, box plots, scatter matrices
- Pollutants analysed: Ozone, CO, SO₂, NO₂ with timestamps and geographic coordinates

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Google Colab

```bash
python "Air Pollution Data Analysis.py"
```

---

## 📈 Part 2 — ARIMA Forecasting (R)

**Goal:** Model and forecast particulate matter trends using ARIMA time-series analysis.

- Stationarity testing (ADF test), ACF/PACF analysis
- First-order differencing; ARIMA(1,0,1) model fitted on training data
- Short-term forecasts generated and validated with Ljung-Box test

**Tools:** R, RStudio, `forecast`, `tseries`

```r
install.packages(c("forecast", "tseries"))
source("TimeSeriesTR.R")
source("TimeSeriesTS.R")
```

---

## 📊 Dataset

- `PollutionData190100.csv` and `PollutionData209960.csv` — real-world Aarhus air quality data
- Update file paths in scripts if not using Google Drive / local default paths

---

## 📬 Contact

**Khunsa Iftikhar**
📧 [khunsaiftikhar123@gmail.com](mailto:khunsaiftikhar123@gmail.com)
🔗 [linkedin.com/in/khunsa-iftikhar](https://www.linkedin.com/in/khunsa-iftikhar/)
