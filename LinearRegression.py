
# %%
import sys
sys.path.insert(0, '/home/youyucai/.local/lib/python3.5/site-packages/fastai')
import fastai
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
df = pd.read_csv('data/NSE-TATAGLOBAL11.csv')

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
# add_datepart(new_data, 'Date')
# new_data.drop('Elapsed', axis=1, inplace=True) # elapsed will be the time stamp

# # more features
# new_data['mon_fri'] = 0
# for i in range(0, len(new_data)):
#     if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
#         new_data['mon_fri'][i] = 1
#     else:
#         new_data['mon_fri'][i] = 0

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

new_data.shape, train.shape, valid.shape
train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()

x_train = train.drop('Close', axis = 1)
y_train = train['Close']
x_valid = valid.drop('Close', axis = 1)
y_valid = valid['Close']

# implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# make predictions and find the rmse
preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)), 2)))
rms

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds
valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])


# %%