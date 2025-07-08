import math
import pytz
from datetime import datetime as dt
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
import yfinance as yf
import sys
plt.style.use('fivethirtyeight')

print(yf.__version__)
# print(pandas_datareader.__version__)
print(sys.version)
print(pd.__version__)

tz = pytz.timezone("Europe/Paris")
# start = tz.localize(dt(2012,1,1))
# end = tz.localize(dt.today())

# tickers = "MA,V,AMZN,JPM,BA".split(",")
tickers = "AAPL"

# download dataframe
df = yf.download(tickers, start="2012-01-01", end="2019-12-17", auto_adjust=True, ignore_tz=False
# ticker = yf.Ticker(tickers)
# df = ticker.history(start="2017-01-01", end="2017-04-30")
print(df)

# Get the number of rows  and colums in the data set
print(df.shape)

# # Visualize the closing price history
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Closed price USD ($)', fontsize=18)
# plt.show()

# Create a new dataframe with only the 'close colum
data = df.filter(['Close'])
# Converte the dataframe to numpy array
dataset = data.values
# Get the number of row to train the model on
training_data_len = math.ceil(len(dataset) * .8)

print(training_data_len)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# Create the training dataset
# Create the scale training dataset

train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train, y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
      print(x_train)
      print(y_train)
      print()

# Convert the x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train de model
rfc = model.fit(x_train, y_train, batch_size=1, epochs=1)

# # predict on test set
# rfc_pred = rfc.predict(x_train)

# # evalute model
# evaluate_model(y_train, rfc_pred)

## Save the model
import joblib
joblib.dump(rfc, "predict_stocks.pkl")

# Create the testing dataset
# Creare a new array contianing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60: :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

# COnvert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the model predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt( np.mean( predictions - y_test )**2 )
rmse

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# # Visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Closed price USD ($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

# Show the valid and predicted prices
print(valid[['Close', 'Predictions']])

valid[['Close', 'Predictions']].to_json('stocks_predictions.json')

print(EventSourceResponse(valid[['Close', 'Predictions']])

## Get the quote
# apple_quote = yf.download(tickers, start="2017-01-01", end="2019-12-17", ignore_tz=True)

# # Create a new dataframe
# new_df = apple_quote.filter(['Close'])
# # Get the lase 60 day closing Price values and convert the dataframe ton an array
# last_60_days = new_df[-60:].values
# # Scale the data to be values between 0 and 60
# last_60_days_scaled = scaler.transform(last_60_days)
# # Create an empty list
# X_test = []
# # Append the past 60 days
# X_test.append(last_60_days_scaled)
# # Convert the X_test data set  to a numpy array
# X_test = np.array(X_test)
# # Reshape the data
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# # Get the predicted sclaed price
# pred_price = model.predict(X_test)
# # undo the scaling
# pred_price = scaler.inverse_transform(pred_price)
# print(pred_price)

# # Get the quote
# apple_quote2 = yf.download(tickers, start="2019-12-17",end="2019-12-18", ignore_tz=True)
# print(apple_quote2['Close'])