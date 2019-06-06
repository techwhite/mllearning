
# %%

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

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
new_data.drop('Date', axis=1, inplace=True)

# preparing data
dataset = new_data.values
train = dataset[0:987,:]
valid = dataset[987:,:]

# converting
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

inputs = new_data[len(new_data)-len(valid)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
x_test, y_test = [], []
for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i, 0])
    y_test.append(inputs[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# model fit
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

from keras import metrics
# optimizer=tf.train.AdamOptimizer(0.001)
# keras.losses=categorical_crossentropy
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.categorical_accuracy])
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, validation_data=(x_test, y_test))

# evaluate
model.evaluate(x_test, y_test)

# make predictions and find the rmse
closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)

rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
print(rms)

# plot
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])

# %%