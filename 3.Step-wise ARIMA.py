import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. Generate sample data
# -----------------------------
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='MS')
sales = 200 + np.random.normal(0, 10, size=100).cumsum()
df = pd.DataFrame({'Date': date_range, 'Sales': sales}).set_index('Date')

# 2. ADF test on original series to determine d

result = adfuller(df['Sales'])
print(" ADF Test on Original Series:")
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
#Since d>1 is not something common we check it only for 1 and 0
if result[1] < 0.05:
    print(" Series is stationary — using d = 0")
    d_values = [0]
else:
    print(" Series is NOT stationary — using d = 1")
    d_values = [1]

# 3. Define ARIMA(p,[0 or1},q) search grid for higher d ,d_values = range(0, x) should be added
p_values = range(0, 4)
q_values = range(0, 4)
orders = list(product(p_values, d_values, q_values))

# 4. Search for best model using ADFand AIC
best_aic = float("inf")
best_model = None
best_order = None

def is_stationary(residuals):
    adf_result = adfuller(residuals)
    return adf_result[1] < 0.05

print("\n Searching ARIMA models...")
for order in orders:
    try:
        model = ARIMA(df['Sales'], order=order)
        model_fit = model.fit()
        residuals = model_fit.resid

        if is_stationary(residuals):
            aic = model_fit.aic
            print(f"ARIMA{order} - AIC: {aic:.2f}  Residuals stationary")
            if aic < best_aic:
                best_aic = aic
                best_model = model_fit
                best_order = order
        else:
            print(f"ARIMA{order} -  Residuals NOT stationary")
    except:
        continue
# 5. Final result and diagnostics
if best_model:
    print("\n Best ARIMA model with stationary residuals:", best_order)
    print("AIC:", best_model.aic)
    print("BIC:", best_model.bic)
else:
    raise ValueError(" No suitable ARIMA model found with stationary residuals.")

# 6. Forecast next 2 months
forecast_steps = 2
forecast = best_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=forecast_steps, freq='MS')
forecast_series = pd.Series(forecast, index=forecast_index)

# 7. Plot forecast
print(f"predicted values is {forecast_series}")
plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_series, label=f'Forecast (ARIMA{best_order})', color='red', marker='o')
plt.title(f"12-Month Forecast using ARIMA{best_order} (Stationary Residuals)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# 8. Plot residuals
residuals = best_model.resid
plt.figure(figsize=(12, 4))
plt.plot(residuals)
plt.title("Residuals Over Time (from Best ARIMA Model)")
plt.grid()
plt.tight_layout()
plt.show()
