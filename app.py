import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn
import joblib
import numpy as np
from pydantic import BaseModel

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
plt.style.use('fivethirtyeight')

class PredictStocks(BaseModel):
    stocks: str

app = FastAPI(
    title="Financial API",
    description="""An API that utilises a Machine Learning model to create json data with stocks and apply a model of prediction""",
    version="1.0.0", debug=True)

def download_stock_data(tickers, start_date="2012-01-01", end_date="2019-12-17"):
    """
    Download stock data from Yahoo Finance
    """
    try:
        df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, ignore_tz=True)
        print("Colonnes du DataFrame:", df.columns)
        
        if df.empty:
            raise ValueError("Aucune donnée téléchargée pour ce ticker et cette période.")
        
        # Correction pour MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            print("Colonnes après aplatissement:", df.columns)
        
        return df
    except Exception as e:
        raise ValueError(f"Erreur lors du téléchargement des données: {str(e)}")

def get_close_data(df, tickers):
    """
    Extract close price data from DataFrame
    """
    # Trouver la colonne 'Close' (insensible à la casse)
    close_cols = [col for col in df.columns if col.lower().startswith('close')]
    if not close_cols:
        raise ValueError(f"Aucune colonne 'Close' trouvée dans les données téléchargées. Colonnes disponibles : {list(df.columns)}")
    
    data = df[close_cols]
    print("DataFrame utilisé pour MinMaxScaler :", data.head())
    print("Shape de data :", data.shape)
    
    if data.empty or data.shape[1] == 0:
        raise ValueError(f"Pas de données 'Close' exploitables pour le(s) ticker(s) {tickers}.")
    
    return data

def prepare_lstm_data(data, training_data_ratio=0.8, sequence_length=60):
    """
    Prepare data for LSTM model training
    """
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * training_data_ratio)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create the training dataset
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    
    for i in range(sequence_length, len(train_data)):
        x_train.append(train_data[i-sequence_length:i, 0])
        y_train.append(train_data[i, 0])
    
    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, training_data_len, scaled_data

def build_lstm_model(input_shape, lstm_units=50, dense_units=25):
    """
    Build LSTM model
    """
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(dense_units))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def make_predictions(model, scaler, scaled_data, training_data_len, sequence_length=60):
    """
    Make predictions using trained model
    """
    # Create the testing dataset
    test_data = scaled_data[training_data_len - sequence_length:, :]
    x_test = []
    
    for i in range(sequence_length, len(test_data)):
        x_test.append(test_data[i-sequence_length:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Get predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions

@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Financial API 🙌🏻

Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
    """
    return note

favicon_path = 'favicon.png'
@app.get('/favicon.png', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

data_path = 'stocks.json'
@app.get('/getdata')
def getdata():
    return FileResponse(data_path)

@app.post('/createdata')
def createdata(data: PredictStocks):
    try:
        features = np.array(data.stocks)
        feat = str(features)
        tickers = feat.split(",")
        
        # Download data
        df = download_stock_data(tickers)
        
        # Convert to JSON
        result = df.to_json(orient="table")
        parsed = loads(result)
        return parsed
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post('/predict')
def predict(data: PredictStocks):
    try:
        print(f"Starting prediction for stocks: {data.stocks}")
        
        features = np.array(data.stocks)
        feat = str(features)
        tickers = feat.split(",")
        print(f"Tickers to process: {tickers}")
        
        # Download and prepare data
        print("Downloading stock data...")
        df = download_stock_data(tickers)
        print(f"Downloaded data shape: {df.shape}")
        
        print("Extracting close data...")
        data = get_close_data(df, tickers)
        print(f"Close data shape: {data.shape}")
        
        # Get the number of rows and columns in the data set
        print(f"Original DataFrame shape: {df.shape}")
        
        # Prepare LSTM data
        print("Preparing LSTM data...")
        x_train, y_train, scaler, training_data_len, scaled_data = prepare_lstm_data(data)
        print(f"Training data shape: x_train={x_train.shape}, y_train={y_train.shape}")
        
        # Build and train model
        print("Building and training LSTM model...")
        model = build_lstm_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
        print("Model training completed")
        
        # Make predictions
        print("Making predictions...")
        predictions = make_predictions(model, scaler, scaled_data, training_data_len)
        print(f"Predictions shape: {predictions.shape}")
        
        # Prepare results
        print("Preparing results...")
        train = data[:training_data_len]
        valid = data[training_data_len:].copy()  # Create a copy to avoid SettingWithCopyWarning
        print(f"Valid data shape before adding predictions: {valid.shape}")
        print(f"Valid data columns: {valid.columns.tolist()}")
        
        # Ensure predictions array matches the valid data length
        if len(predictions) != len(valid):
            print(f"Warning: predictions length ({len(predictions)}) != valid length ({len(valid)})")
            # Truncate or pad predictions to match valid length
            if len(predictions) > len(valid):
                predictions = predictions[:len(valid)]
            else:
                # Pad with the last prediction value
                last_pred = predictions[-1] if len(predictions) > 0 else 0
                padding = np.full(len(valid) - len(predictions), last_pred)
                predictions = np.concatenate([predictions, padding])
        
        valid['Predictions'] = predictions
        print(f"Valid data shape after adding predictions: {valid.shape}")
        
        # Return results
        print("Converting to JSON...")
        # Get the first column (which should be the close price column)
        close_column = valid.columns[0]
        result = valid[[close_column, 'Predictions']].to_json(orient="table")
        parsed = loads(result)
        print("Prediction completed successfully")
        return parsed
        
    except ValueError as e:
        print(f"ValueError in predict: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error in predict: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)