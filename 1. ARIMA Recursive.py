import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# 1. Generate fake sales data ,this section should be replaced with actual data
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
sales = 200 + np.random.normal(0, 10, size=100).cumsum()  # Simulated trend
df = pd.DataFrame({'Date': date_range, 'Sales': sales}).set_index('Date')

# 2. Plot sales data
df['Sales'].plot(title='Monthly Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# 3. Check stationarity using Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    return result[1] <= 0.05  # Stationary if p-value <= 0.05

is_stationary = check_stationarity(df['Sales'])

# 4. Differencing if non-stationary
if not is_stationary:
    df['Sales_diff'] = df['Sales'].diff().dropna()
    is_stationary = check_stationarity(df['Sales_diff'].dropna())

# 5. Fit ARIMA model (order can be tuned), auto ARIMA has not implimented since past analysis might alerady show specific order
model = ARIMA(df['Sales'], order=(1, 1, 1))  # (p,d,q)
model_fit = model.fit()
print(model_fit.summary())

# 6. Forecast next 2 months
forecast = model_fit.forecast(steps=2)
forecast_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=2, freq='M')
forecast_df = pd.Series(forecast, index=forecast_dates)

# 7. Plot actual + forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_df, label='Forecast', color='red')
plt.title('Sales Forecast with ARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
