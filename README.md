# 📊 Univariate Time Series Forecasting 

This project implements different methods for forecasting future values using **univariate time series models**.

---

## 📈 ARIMA

The **ARIMA (AutoRegressive Integrated Moving Average)** model is used to forecast  output based on historical data. ARIMA is one of the simplest yet effective methods for analyzing and predicting time series in many practical scenarios.
While ARIMA models are reliable for **short-term forecasting**, their accuracy often declines over longer horizons. Therefore, they are best suited for cases where **near-future predictions** are the goal.

---

## 🔧 Features

- 📅 Simulated sales data
- ✅ Augmented Dickey-Fuller (ADF) test for stationarity
- 🔁 Differencing when required to achieve stationarity
- 🔍 Visualizations to understand trends and forecasts
- 🔢 Direct and recursive multi-step ARIMA forecasting
- 📉 Model evaluation using AIC and BIC

---
## ❄️  SARIMA
In many real-world cases, a time series is influenced not only by its past values but also by specific times or seasons.
For example, consider the sales of winter clothing—it's natural to expect an increase in sales as winter approaches.
This kind of pattern reflects seasonality, which standard ARIMA models (ARIMA(p,d,q)) don’t capture on their own.

To handle this, Seasonal ARIMA (SARIMA) models—often written as ARIMA(p,d,q)(P,D,Q)[s]—include additional components to account for seasonal behavior:

P: Number of seasonal autoregressive (AR) terms

D: Number of seasonal differences

Q: Number of seasonal moving average (MA) terms

s: The number of time steps in a season (e.g., 12 for monthly data with yearly seasonality)

These seasonal parameters help the model better capture patterns that repeat over fixed periods, improving forecast accuracy in data with strong seasonal effects.

---
## 🚧 In Progress / Upcoming Models

This project is under active development. Planned future models include:

- 🎲 **Monte Carlo Simulation** – for probabilistic forecasting and risk estimation
- 📉 **ARCH/GARCH Models** – for modeling volatility and heteroskedasticity
- 🌲 **Random Forest Regression** – for non-linear forecasting and feature-based models
- 🧠 **Neural Networks (e.g., LSTM)** – for capturing complex, non-linear patterns

Stay tuned as these models are added and compared with ARIMA for forecasting performance.
