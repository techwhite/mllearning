
# %%

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

# for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

# read the file
df = pd.read_csv('~/dev/src/mllearning/data/NSE-TATAGLOBAL11.csv')

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

# creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
new_data.index = new_data['Date']

# preparing data
new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

# importing prophet
from fbprophet import Prophet
model = Prophet()
model.fit(train)

# predictions
close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)

print(close_prices)
print(forecast)

# make predictions and find the rmse
forecast_valid = forecast['yhat'][987:]
rms = np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)), 2)))
print(forecast_valid)
print(rms)

# plot
valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])

# %%