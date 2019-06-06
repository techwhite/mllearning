
# %%

import pandas as pd
import numpy as np

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# read the file
df = pd.read_csv('~/dev/src/mllearning/data/NSE-TATAGLOBAL11.csv')

# setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

# creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)

train = data[:987]
valid = data[987:]

training = train['Close']
validation = valid['Close']

print(training)

# fit the model and make predictions
from pyramid.arima import auto_arima
model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0,
seasonal=True,d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True)
print("fiting...")
model.fit(training)

forecast = model.predict(n_periods=248)

print(forecast)

forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

rms = np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])), 2)))
print(rms)

#plot
plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(forecast['Prediction'])


# %%