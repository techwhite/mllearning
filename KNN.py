
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
df = pd.read_csv('~/dev/src/hello/data/NSE-TATAGLOBAL11.csv')

# setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

# creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# # create features
from fastai.structured import add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True) # elapsed will be the time stamp

# # more features
new_data['mon_fri'] = 0
for i in range(0, len(new_data)):
   if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
      new_data['mon_fri'][i] = 1
   else:
      new_data['mon_fri'][i] = 0

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis = 1)
y_train = train['Close']
x_valid = valid.drop('Close', axis = 1)
y_valid = valid['Close']

# importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

# using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# fit the model and make predictions
model.fit(x_train, y_train)
preds = model.predict(x_valid)

rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)), 2)))
rms

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])


# %%