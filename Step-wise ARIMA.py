import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Simulated sales data 
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
sales = 200 + np.random.normal(0, 10, size=100).cumsum()
df = pd.DataFrame({'Date': date_range, 'Sales': sales}).set_index('Date')

# Use auto_arima to find the best ARIMA
stepwise_model = auto_arima(
    df['Sales'],
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    seasonal=False,
    stepwise=True,
    trace=True,     # Prints the steps
    suppress_warnings=True,
    information_criterion='aic'  # Can also use 'bic'
)

print(stepwise_model.summary())

# Fit ARIMA model with selected parameters
best_order = stepwise_model.order
model = ARIMA(df['Sales'], order=best_order)
model_fit = model.fit()
print(model_fit.summary())

# Forecast next month
forecast = model_fit.forecast(steps=1)
forecast_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=1, freq='M')
forecast_df = pd.Series(forecast, index=forecast_dates)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_df, label='Forecast', color='red')
plt.title(f'Sales Forecast with ARIMA{best_order}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
