import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import scipy.stats as stats
from itertools import product
import warnings
warnings.filterwarnings("ignore")
# 1. generate fake time series/for illustration, only use real data
np.random.seed(42)
n = 120
seasonal_pattern = np.sin(np.arange(n) * 2 * np.pi / 12) * 20
trend = np.linspace(100, 200, n)
noise = np.random.normal(0, 5, n)
sales = trend + seasonal_pattern + noise
dates = pd.date_range(start='2013-01-01', periods=n, freq='MS')
series = pd.Series(sales, index=dates)

# 2. Define SARIMA range
p = d = q = range(0, 2)
P = D = Q = range(0, 2)
s = 12
pdq = list(product(p, d, q))
seasonal_pdq = list(product(P, D, Q))
# 3. Finding best model for SARIMA
best_aic = float("inf")
best_model = None
best_order = None
best_seasonal = None
print(" Searching for best SARIMA model :")
for order in pdq:
    for seasonal_order in seasonal_pdq:
        try:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order + (s,))
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_model = results
                best_order = order
                best_seasonal = seasonal_order
            print(f"SARIMA{order}x{seasonal_order + (s,)} - AIC: {results.aic:.2f}")
        except:
            continue
#  Show best model
print("\n Best SARIMA model:")
print("Order:", best_order)
print("Seasonal:", best_seasonal + (s,))
print("AIC:", best_model.aic)
print("BIC:", best_model.bic)

#  Forecast
forecast = best_model.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
forecast_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

#  Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(series, label='Historical Data')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title(f"SARIMA Forecast {best_order} x {best_seasonal + (s,)}")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
#  Residual analysis
residuals = best_model.resid
# ADF test
print("\n📉 ADF Test on Residuals:")
adf_result = adfuller(residuals)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
# Ljung-Box test
print("\n📊 Ljung-Box test (lag=10):")
lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_result)
# Plot residuals
plt.figure(figsize=(12, 4))
plt.plot(residuals)
plt.title("Residuals Over Time")
plt.grid()
plt.tight_layout()
plt.show()
# Histogram + Q-Q Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True)
plt.title("Histogram of Residuals")
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.show()
# ACF of residuals
plt.figure(constrained_layout=True)
plot_acf(residuals, lags=15)
plt.title("ACF of Residuals")
plt.grid()
plt.tight_layout()
plt.show()
