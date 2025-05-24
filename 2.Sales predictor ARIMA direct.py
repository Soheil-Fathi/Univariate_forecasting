import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Generate fake sales data, Replace with true data it is for illustration 
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
sales = 200 + np.random.normal(0, 10, size=100).cumsum()
df = pd.DataFrame({'Date': date_range, 'Sales': sales}).set_index('Date')

# Forecast horizon (e.g., predict t+1 and t+2)
H = 2
for h in range(1, H+1):
    df[f't+{h}'] = df['Sales'].shift(-h)
df_model = df.dropna()
direct_forecasts = {}
train_X = df_model['Sales']
for h in range(1, H+1):
    # Target = future value at horizon h
    target_y = df_model[f't+{h}']
    model = ARIMA(train_X, order=(1, 1, 1))  # Replace with tuned order if needed
    model_fit = model.fit()
    y_pred = model_fit.forecast(steps=h)[-1]
    future_date = df.index[-1] + pd.DateOffset(months=h)
    direct_forecasts[future_date] = y_pred
forecast_series = pd.Series(direct_forecasts)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_series, label='Direct ARIMA Forecast', marker='o', color='green')
plt.title(f'ARIMA Direct Forecast ({H} steps ahead)')
plt.xlabel('Date')
plt.ylabel('Sales')
print(forecast_series)
plt.legend()
plt.grid()
plt.show()
