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
from json import loads, dumps
from sse_starlette.sse import EventSourceResponse
plt.style.use('fivethirtyeight')

print(yf.__version__)
print(sys.version)

tickers = "AAPL"

# download dataframe
df = yf.download(tickers, start="2012-01-01", end="2019-12-17", auto_adjust=True, ignore_tz=True)
print(df)

# Get the number of rows and colums in the data set
dfshape = df.shape
result = df.to_json(orient="table")
parsed = loads(result)
print(dumps(parsed, indent=4))

# print(EventSourceResponse(result))
