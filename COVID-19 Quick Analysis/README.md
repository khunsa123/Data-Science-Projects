# 🦠 COVID-19 Quick Analysis

## 📌 Overview

A quick **exploratory data analysis (EDA)** of global COVID-19 trends using the publicly available Our World in Data (OWID) dataset, streamed directly from GitHub. The analysis focuses on daily new case trends for Canada and a global comparison of total cases across the most affected countries.

---

## 🎯 Objectives

- Load and process live COVID-19 data from the OWID dataset
- Visualise daily new case trends for Canada over the last 60 days
- Identify and compare the top 5 countries by total confirmed cases

---

## 🧠 Methods

- Live data ingestion from the [OWID COVID-19 GitHub repository](https://github.com/owid/covid-19-data)
- Data filtering and time-series slicing with Pandas
- Trend visualisation with Seaborn and Matplotlib

---

## 📊 Dataset

- **Source:** Our World in Data (OWID) — `owid-covid-data.csv`
- **Access:** Loaded directly via URL — no manual download required
- **Key columns used:** `location`, `date`, `total_cases`, `new_cases`, `total_deaths`, `new_deaths`

---

## 📈 Outputs

| Plot | Description |
|------|-------------|
| `canada_new_cases.png` | Daily new COVID-19 cases in Canada (last 60 days) |
| `top5_countries_cases.png` | Bar chart of top 5 countries by total confirmed cases |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, Matplotlib, Seaborn

---

## ⚙️ Usage

```bash
pip install pandas matplotlib seaborn
python covid_analysis.py
```

---

## 📬 Contact

**Khunsa Iftikhar**
📧 [khunsaiftikhar123@gmail.com](mailto:khunsaiftikhar123@gmail.com)
🔗 [linkedin.com/in/khunsa-iftikhar](https://www.linkedin.com/in/khunsa-iftikhar/)
