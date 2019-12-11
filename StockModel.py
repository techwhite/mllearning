
# %%

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import metrics

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

# tuning params
back_step = 60
num_unit = 500
epoch = 1
batch_size = 1
dropout_rate = 0.2
sample_rate = 0.8

# read the file
df = pd.read_csv('~/dev/src/mllearning/data/worldtradingdata-history-BABA.csv')

# setting index as date
df.index = df['Date']

# creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = data.copy(deep=True)
new_data.drop('Date', axis=1, inplace=True)
new_data.reset_index(drop=True, inplace=True)

# preparing data
# or new_data.values
dataset = new_data.to_numpy()
train_len = (int)(len(dataset) * sample_rate)
train = dataset[0:train_len,:]
valid = dataset[train_len:,:]

# converting
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(back_step, len(train)):
    x_train.append(scaled_data[i-back_step:i])
    y_train.append(scaled_data[i,1])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
print(x_train.shape)
print(y_train.shape)

inputs = new_data[len(new_data)-len(valid)-back_step:].values
inputs = scaler.transform(inputs)
x_test, y_test = [], []
for i in range(back_step, inputs.shape[0]):
    x_test.append(inputs[i-back_step:i])
    y_test.append(inputs[i, 1])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
print(x_test.shape)
print(y_test.shape)

# model fit
model = Sequential()
model.add(LSTM(units=num_unit, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(dropout_rate))
model.add(LSTM(units=num_unit))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

# optimizer=tf.train.AdamOptimizer(0.001)
# keras.losses=categorical_crossentropy
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.categorical_accuracy])
model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=2, validation_split=0.25)

# summary
model.summary()

# evaluate
model.evaluate(x_test, y_test)

# make predictions and find the rmse
closing_price = model.predict(x_test)

# scaler saved information about n features. Need recreated another to do inverse_transform
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler1.min_, scaler1.scale_ = scaler.min_[1], scaler.scale_[1]
closing_price = scaler1.inverse_transform(closing_price)

rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
print(rms)

# plot
train = new_data[:train_len]
valid = new_data[train_len:]
valid['Predictions'] = closing_price

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])

print(valid[['Close', 'Predictions']])

# %%