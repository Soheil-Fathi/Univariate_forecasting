# 📊 Univariate Time Series Forecasting 

This project implements different methods for forecasting future values using **univariate time series models**.

---

## 📈 Sales Forecasting using ARIMA

The **ARIMA (AutoRegressive Integrated Moving Average)** model is used to forecast  output based on historical data. ARIMA is one of the simplest yet effective methods for analyzing and predicting time series in many practical scenarios.
While ARIMA models are reliable for **short-term forecasting**, their accuracy often declines over longer horizons. Therefore, they are best suited for cases where **near-future predictions** are the goal.

---

## 🔧 Features

- 📅 Simulated monthly sales data
- ✅ Augmented Dickey-Fuller (ADF) test for stationarity
- 🔁 Differencing when required to achieve stationarity
- 🔍 Visualizations to understand trends and forecasts
- 🔢 Direct and recursive multi-step ARIMA forecasting
- 📉 Model evaluation using AIC

---
## 🚧 In Progress / Upcoming Models

This project is under active development. Planned future models include:

- ❄️ **Seasonal ARIMA (SARIMA)** – to capture seasonality in the data
- 🎲 **Monte Carlo Simulation** – for probabilistic forecasting and risk estimation
- 📉 **ARCH/GARCH Models** – for modeling volatility and heteroskedasticity
- 🧠 **Neural Networks (e.g., LSTM)** – for capturing complex, non-linear patterns

Stay tuned as these models are added and compared with ARIMA for forecasting performance.
