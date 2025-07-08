from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import datetime
import time
import sys

print(yf.__version__) # '0.1.87'
# print(pandas_datareader.__version__) # '0.10.0'
print(sys.version) # 3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]


# df.to_csv(f'{ticker}.csv')

# download dataframe
# data = yf.download('AAPL', start="2017-01-01", end="2017-04-30", ignore_tz=True)
# print(data)

# df = pdr.get_data_yahoo('AAPL', start="2017-01-01", end="2017-04-30")
# print(df)

ticker = yf.Ticker('AAPL')
todays_data = ticker.history(start="2017-01-01", end="2017-04-30")
print(todays_data)