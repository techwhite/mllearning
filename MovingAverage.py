
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
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

new_data.shape, train.shape, valid.shape

train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()

# make predictions
preds = []
for i in range(0,248):
    a = train['Close'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

# calculate rmse
convertedarray = (np.power((np.array(valid['Close'])-preds), 2)).astype(float)
rms = np.sqrt(np.mean(convertedarray))
rms

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])


# %%