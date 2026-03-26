# 📈 Air Pollution Time-Series Forecasting (R)

## 📌 Overview
This project focuses on **time-series analysis and forecasting of air pollution data** using the **ARIMA model in R**. It extends previous exploratory analysis by modeling and predicting pollutant trends, specifically **particulate matter**, from real-world datasets.

---

## 🎯 Objectives
- Apply **time-series modeling (ARIMA)** on pollution data  
- Perform **stationarity testing** using ADF test  
- Analyze temporal patterns using **ACF and PACF**  
- Generate short-term **forecasts for air quality trends**  

---

## 📊 Dataset
- `pollutionData190100.csv`  
- `pollutionData209960.csv`  

- Contains environmental attributes such as particulate matter and timestamps  
- Used for both training and testing time-series models  

---

## 🛠️ Methods & Techniques

### 🔹 Data Preparation
- Converted data into time-series format using `ts()`  
- Split into **training and testing sets**  

### 🔹 ARIMA Modeling
- Applied **first-order differencing (d = 1)**  
- Identified parameters using:
  - **ACF (Auto-Correlation Function)**  
  - **PACF (Partial Auto-Correlation Function)**  
- Built ARIMA model: **ARIMA(1,0,1)**  

### 🔹 Forecasting & Evaluation
- Generated forecasts using `forecast()`  
- Residual analysis:
  - ACF of residuals  
  - **Ljung-Box test** for model validation  

---

## 📂 Project Files

│── TimeSeriesTR.R   # Training data modeling & forecasting

│── TimeSeriesTS.R   # Testing data modeling & forecasting

│── pollutionData190100.csv

│── pollutionData209960.csv

│── README.md


---

## 🧰 Tools & Libraries

- **Language:** R
- **Environment:** RStudio
- **Libraries:** 
  - forecast  
  - tseries  

---

## 🚀 How to Run

1. Open RStudio
2. Set working directory to project folder
3. Install required packages: 
  - install.packages("forecast") 
  - install.packages("tseries") 
4. Run scripts: 
  - source("TimeSeriesTR.R") 
  - source("TimeSeriesTS.R")   

---

## 📈 Key Outcomes

- Successfully modeled pollution data using ARIMA 
- Identified temporal dependencies in particulate matter levels
- Generated short-term forecasts for air quality trends
- Validated model performance using statistical tests

---

## ⚠️ Notes

- Update file paths in .R scripts according to your system 
- Results may vary depending on dataset size and preprocessing

________________________________________
💡 This project demonstrates the application of time-series modeling in environmental data analysis and forecasting.
